from torch import nn
import torch

class TensorFlowBatchNorm1d(nn.BatchNorm1d):
    """当前 CLSR 使用的二维 BatchNorm，按 TF1 非融合路径保存总体方差。"""

    def __init__(self, num_features, eps=1e-4, momentum=0.05):
        super().__init__(num_features, eps=eps, momentum=momentum)

    def forward(self, inputs):
        if inputs.ndim != 2:
            raise ValueError("这里的 BatchNorm 输入必须是 [位置数量, 特征维度]")

        if self.training:
            # 归一化和运行方差都使用总体方差，即除以 N。
            variance, mean = torch.var_mean(
                inputs, dim=0, unbiased=False
            )

            with torch.no_grad():
                self.num_batches_tracked.add_(1)
                self.running_mean.mul_(1.0 - self.momentum)
                self.running_mean.add_(mean.detach(), alpha=self.momentum)
                self.running_var.mul_(1.0 - self.momentum)
                self.running_var.add_(variance.detach(), alpha=self.momentum)
        else:
            mean = self.running_mean
            variance = self.running_var

        normalized = (inputs - mean) * torch.rsqrt(variance + self.eps)
        return normalized * self.weight + self.bias


class TensorFlowEmbedding(nn.Embedding):
    """保留每次查询的原始梯度贡献，供裁剪后再按 ID 合并。"""

    def __init__(self, num_embeddings, embedding_dim):
        super().__init__(num_embeddings, embedding_dim)
        self.gradient_parts = []

    def clear_gradient_parts(self):
        self.gradient_parts.clear()

    def forward(self, indices):
        embeddings = super().forward(indices)

        if torch.is_grad_enabled() and embeddings.requires_grad:
            flat_indices = indices.detach().reshape(-1)

            def record_gradient(gradient):
                # 同一查询结果的所有用途，先在这个结果上汇总梯度。
                # 不同查询的贡献保留为不同部分，尚不按 ID 合并。
                values = gradient.detach().reshape(-1, self.embedding_dim).clone()
                self.gradient_parts.append((flat_indices, values))

            embeddings.register_hook(record_gradient)

        return embeddings



class Time4LSTMCell(nn.Module):
    """按官方源码逐步实现带时间门的 LSTM 单元"""

    def __init__(self, input_size=40, hidden_size=40):
        super().__init__()

        self.hidden_size = hidden_size

        # 当前商品表示和上一步输出拼接后，一次计算四组特征
        self.lstm_gates = nn.Linear(
            input_size + hidden_size,
            hidden_size * 4,
        )

        # 对应官方这一层的默认初始化
        nn.init.xavier_uniform_(self.lstm_gates.weight)
        nn.init.zeros_(self.lstm_gates.bias)

        # 把距离目标交互的时间转换成隐藏维度的表示
        self.time_input_w1 = nn.Parameter(torch.empty(hidden_size))
        self.time_input_bias1 = nn.Parameter(torch.empty(hidden_size))

        # 把从首次行为起经过的时间转换成隐藏维度的表示
        self.time_input_w2 = nn.Parameter(torch.empty(hidden_size))
        self.time_input_bias2 = nn.Parameter(torch.empty(hidden_size))

        # 对应官方这四个一维参数的默认 Glorot 均匀初始化
        init_bound = (3.0 / hidden_size) ** 0.5

        for parameter in [
            self.time_input_w1,
            self.time_input_bias1,
            self.time_input_w2,
            self.time_input_bias2,
        ]:
            nn.init.uniform_(parameter, -init_bound, init_bound)

        # 距离目标的时间门：商品变换、时间变换、偏置
        self.time_kernel_w1 = nn.Parameter(
            torch.empty(input_size, hidden_size)
        )
        self.time_kernel_t1 = nn.Parameter(
            torch.empty(hidden_size, hidden_size)
        )
        self.time_bias1 = nn.Parameter(torch.empty(hidden_size))

        # 从首次行为起经过时间的时间门
        self.time_kernel_w2 = nn.Parameter(
            torch.empty(input_size, hidden_size)
        )
        self.time_kernel_t2 = nn.Parameter(
            torch.empty(hidden_size, hidden_size)
        )
        self.time_bias2 = nn.Parameter(torch.empty(hidden_size))

        # 对应官方这些参数的默认初始化
        for parameter in [
            self.time_kernel_w1,
            self.time_kernel_t1,
            self.time_kernel_w2,
            self.time_kernel_t2,
        ]:
            nn.init.xavier_uniform_(parameter)

        for parameter in [self.time_bias1, self.time_bias2]:
            nn.init.uniform_(parameter, -init_bound, init_bound)

        # 两组时间表示对输出门的影响
        self.o_kernel_t1 = nn.Parameter(
            torch.empty(hidden_size, hidden_size)
        )
        self.o_kernel_t2 = nn.Parameter(
            torch.empty(hidden_size, hidden_size)
        )

        nn.init.xavier_uniform_(self.o_kernel_t1)
        nn.init.xavier_uniform_(self.o_kernel_t2)

    def forward(self, inputs, state):
        """读取一次历史行为，更新记忆并返回当前输出"""

        # inputs 包含 40 维商品表示和两个时间特征
        item_input = inputs[:, :-2]
        time_last = inputs[:, -2:-1]
        time_now = inputs[:, -1:]

        # LSTM 的两个状态：内部记忆、上一步输出
        previous_memory, previous_hidden = state

        # 1. 根据当前商品和上一步输出，计算四组原始特征
        (
            input_feature,
            candidate_feature,
            forget_feature,
            output_feature,
        ) = self.calculate_basic_features(
            item_input,
            previous_hidden,
        )

        # 2. 把两个时间特征转换成向量
        time_last_input, time_now_input = self.encode_time(
            time_last,
            time_now,
        )

        # 3. 结合商品表示，计算两个时间门
        time_last_gate, time_now_gate = self.calculate_time_gates(
            item_input,
            time_last_input,
            time_now_input,
        )

        # 4. 计算输入门、遗忘门和候选记忆
        input_gate = torch.sigmoid(input_feature)

        # 官方的 forget_bias 默认为 1.0
        forget_gate = torch.sigmoid(forget_feature + 1.0)

        candidate_memory = torch.tanh(candidate_feature)

        # 5. 保留部分旧记忆，写入部分新内容
        retained_memory = (
            forget_gate * time_last_gate * previous_memory
        )
        written_memory = (
            input_gate * time_now_gate * candidate_memory
        )

        new_memory = retained_memory + written_memory

        # 6. 两组时间表示也参与输出门的计算
        output_feature = (
            output_feature
            + torch.matmul(time_now_input, self.o_kernel_t1)
            + torch.matmul(time_last_input, self.o_kernel_t2)
        )

        output_gate = torch.sigmoid(output_feature)

        # 7. 根据更新后的记忆，生成当前输出
        new_hidden = output_gate * torch.tanh(new_memory)

        return new_hidden, (new_memory, new_hidden)

    def calculate_basic_features(self, item_input, previous_hidden):
        """计算输入门、候选记忆、遗忘门、输出门的原始特征"""

        combined_input = torch.cat(
            [item_input, previous_hidden],
            dim=-1,
        )

        gate_features = self.lstm_gates(combined_input)

        # 官方源码的拆分顺序是 i、j、f、o
        input_feature, candidate_feature, forget_feature, output_feature = (
            gate_features.chunk(4, dim=-1)
        )

        return (
            input_feature,
            candidate_feature,
            forget_feature,
            output_feature,
        )

    def encode_time(self, time_last, time_now):
        """把两组时间特征分别转换成隐藏维度的向量"""

        time_now_input = torch.tanh(
            time_now * self.time_input_w1 + self.time_input_bias1
        )

        time_last_input = torch.tanh(
            time_last * self.time_input_w2 + self.time_input_bias2
        )

        return time_last_input, time_now_input

    def calculate_time_gates(
        self,
        item_input,
        time_last_input,
        time_now_input,
    ):
        """结合当前商品和时间表示，计算两个时间门"""

        time_now_state = (
            torch.matmul(item_input, self.time_kernel_w1)
            + torch.matmul(time_now_input, self.time_kernel_t1)
            + self.time_bias1
        )

        time_last_state = (
            torch.matmul(item_input, self.time_kernel_w2)
            + torch.matmul(time_last_input, self.time_kernel_t2)
            + self.time_bias2
        )

        time_now_gate = torch.sigmoid(time_now_state)
        time_last_gate = torch.sigmoid(time_last_state)

        return time_last_gate, time_now_gate


class CLSRModel(nn.Module):
    def __init__(self, user_vocab_length, item_vocab_length, cate_vocab_length):
        super().__init__()

        # 官方配置中的 embedding 维度。
        self.item_embedding_dim = 32
        self.cate_embedding_dim = 8
        self.user_embedding_dim = 40

        # 官方父类创建的用户、商品和类目表。
        self.user_lookup = TensorFlowEmbedding(
            user_vocab_length, self.user_embedding_dim
        )
        self.item_lookup = TensorFlowEmbedding(
            item_vocab_length, self.item_embedding_dim
        )
        self.cate_lookup = TensorFlowEmbedding(
            cate_vocab_length, self.cate_embedding_dim
        )

        # CLSR 为长期、短期兴趣分别创建的用户表。
        self.user_long_lookup = TensorFlowEmbedding(
            user_vocab_length, self.user_embedding_dim
        )
        self.user_short_lookup = TensorFlowEmbedding(
            user_vocab_length, self.user_embedding_dim
        )

        # 对齐官方截断正态初始化：标准差 0.01，截断在 ±0.02。
        embedding_tables = [
            self.user_lookup,
            self.item_lookup,
            self.cate_lookup,
            self.user_long_lookup,
            self.user_short_lookup,
        ]

        for table in embedding_tables:
            nn.init.trunc_normal_(
                table.weight,
                mean=0.0,
                std=0.01,
                a=-0.02,
                b=0.02,
            )

        # 将历史物品向量变换到与长期 Query 匹配的空间。
        # 官方这里只有变换矩阵，所以关闭偏置。
        self.long_attention_projection = nn.Linear(
            40, 40, bias=False
        )

        nn.init.trunc_normal_(
            self.long_attention_projection.weight,
            mean=0.0,
            std=0.01,
            a=-0.02,
            b=0.02,
        )

        # 每个历史位置的 160 维特征，最终变成一个注意力分数。
        self.long_attention_mlp = nn.Sequential(
            nn.Linear(160, 80),
            TensorFlowBatchNorm1d(80, eps=0.0001, momentum=0.05),
            nn.ReLU(),

            nn.Linear(80, 40),
            TensorFlowBatchNorm1d(40, eps=0.0001, momentum=0.05),
            nn.ReLU(),

            nn.Linear(40, 1),
        )

        # 按官方规则初始化线性层的权重和偏置。
        for layer in self.long_attention_mlp:
            if isinstance(layer, nn.Linear):
                nn.init.trunc_normal_(
                    layer.weight,
                    mean=0.0,
                    std=0.01,
                    a=-0.02,
                    b=0.02,
                )
                nn.init.zeros_(layer.bias)

        # 短期兴趣演化 GRU，所有历史位置共用这一套参数。
        self.short_evolution_gru = TensorFlowGRUCell(
            input_size=40,
            hidden_size=40,
        )

        # 对历史行为逐条编码，参数在整条历史中共享
        self.short_history_lstm = Time4LSTMCell(
            input_size=40,
            hidden_size=40,
        )

        # 将历史输出从 40 维映射到短期 Query 的 80 维
        self.short_attention_projection = nn.Linear(
            40,
            80,
            bias=False,
        )

        nn.init.trunc_normal_(
            self.short_attention_projection.weight,
            mean=0.0,
            std=0.01,
            a=-0.02,
            b=0.02,
        )

        # 将每个历史位置的 320 维特征转换成一个分数
        self.short_attention_mlp = nn.Sequential(
            nn.Linear(320, 80),
            TensorFlowBatchNorm1d(80, eps=1e-4, momentum=0.05),
            nn.ReLU(),

            nn.Linear(80, 40),
            TensorFlowBatchNorm1d(40, eps=1e-4, momentum=0.05),
            nn.ReLU(),

            nn.Linear(40, 1),
        )

        for layer in self.short_attention_mlp:
            if isinstance(layer, nn.Linear):
                nn.init.trunc_normal_(
                    layer.weight,
                    mean=0.0,
                    std=0.01,
                    a=-0.02,
                    b=0.02,
                )
                nn.init.zeros_(layer.bias)

        # 独立的融合 GRU，与短期意图 GRU 不共享参数
        self.fusion_gru = TensorFlowGRUCell(
            input_size=40,
            hidden_size=40,
        )

        # 根据历史、目标商品和长短期兴趣预测融合比例
        self.fusion_mlp = nn.Sequential(
            nn.Linear(161, 80),
            TensorFlowBatchNorm1d(80, eps=1e-4, momentum=0.05),
            nn.ReLU(),

            nn.Linear(80, 40),
            TensorFlowBatchNorm1d(40, eps=1e-4, momentum=0.05),
            nn.ReLU(),

            nn.Linear(40, 1),
        )

        for layer in self.fusion_mlp:
            if isinstance(layer, nn.Linear):
                nn.init.trunc_normal_(
                    layer.weight,
                    mean=0.0,
                    std=0.01,
                    a=-0.02,
                    b=0.02,
                )
                nn.init.zeros_(layer.bias)

        # 根据融合兴趣和目标商品计算预测分数
        self.prediction_mlp = nn.Sequential(
            nn.Linear(80, 100),
            TensorFlowBatchNorm1d(100, eps=1e-4, momentum=0.05),
            nn.ReLU(),

            nn.Linear(100, 64),
            TensorFlowBatchNorm1d(64, eps=1e-4, momentum=0.05),
            nn.ReLU(),

            nn.Linear(64, 1),
        )

        for layer in self.prediction_mlp:
            if isinstance(layer, nn.Linear):
                nn.init.trunc_normal_(
                    layer.weight,
                    mean=0.0,
                    std=0.01,
                    a=-0.02,
                    b=0.02,
                )
                nn.init.zeros_(layer.bias)

    def forward(self, batch):
        """完成从输入批次到预测分数的整条计算"""

        # 1. 查询并拼接 Embedding
        embeddings = self.lookup_embeddings(batch)

        hist_input = embeddings["hist_input"]
        target_item_embedding = embeddings["target_item_embedding"]
        mask = batch["mask"]

        # 计算对比学习需要的两个兴趣代理
        hist_mean, hist_recent = self.build_interest_proxies(
            hist_input,
            mask,
        )

        # 2. 编码长期兴趣
        long_interest, _ = self.encode_long_term(
            embeddings["user_long_embedding"],
            hist_input,
            mask,
        )

        # 3. 计算短期意图，作为短期 Query 的组成部分
        short_term_intention = self.encode_short_intention(
            hist_input,
            embeddings["user_short_embedding"],
            mask,
        )

        # 4. 用 Time4LSTM 编码整条历史
        short_history_input = self.build_short_history_input(
            hist_input,
            batch,
        )

        rnn_outputs = self.encode_short_history(
            short_history_input,
            mask,
        )

        # 5. 通过短期注意力得到短期兴趣
        short_interest, _ = self.encode_short_term(
            short_term_intention,
            target_item_embedding,
            rnn_outputs,
            mask,
        )

        # 6. 计算融合所需的历史状态
        fusion_state = self.encode_fusion_history(
            hist_input,
            mask,
        )

        # 7. 自适应融合长短期兴趣
        fused_interest, alpha = self.fuse_interests(
            fusion_state,
            target_item_embedding,
            long_interest,
            short_interest,
            batch["time_to_now"],
        )

        # 8. 将融合兴趣与目标商品拼接，计算预测分数
        prediction_input = torch.cat(
            [fused_interest, target_item_embedding],
            dim=-1,
        )

        logits = self.prediction_mlp(prediction_input)
        predictions = torch.sigmoid(logits)

        # 保留长短期兴趣，后面计算辅助损失时还要使用
        return {
            "logits": logits,
            "predictions": predictions,
            "long_interest": long_interest,
            "short_interest": short_interest,
            "alpha": alpha,
            "hist_mean": hist_mean,
            "hist_recent": hist_recent,
        }


    def compute_loss(self, outputs, batch, num_negatives=4):
        """组合四项损失，并保留各项数值供训练时观察"""

        data_loss = self.compute_data_loss(
            outputs,
            batch,
            num_negatives=num_negatives,
        )

        # 官方在正则项和差异损失中共用同一份去重用户查询结果。
        involved_embeddings = self.lookup_involved_embeddings(batch)

        regular_loss = self.compute_regular_loss(
            batch,
            involved_embeddings=involved_embeddings,
        )

        contrastive_loss = self.compute_contrastive_loss(
            outputs,
            batch,
        )

        discrepancy_loss = self.compute_discrepancy_loss(
            batch,
            involved_embeddings=involved_embeddings,
        )

        total_loss = (
            data_loss
            + regular_loss
            + contrastive_loss
            + discrepancy_loss
        )

        return {
            "loss": total_loss,
            "data_loss": data_loss,
            "regular_loss": regular_loss,
            "contrastive_loss": contrastive_loss,
            "discrepancy_loss": discrepancy_loss,
        }


    def lookup_involved_embeddings(self, batch):
        """查询本批去重后的 Embedding，供多个损失共同使用。"""

        involved_items = torch.unique(
            torch.cat([
                batch["item_history"].reshape(-1),
                batch["items"].reshape(-1),
            ])
        )
        involved_cates = torch.unique(
            torch.cat([
                batch["item_cate_history"].reshape(-1),
                batch["cates"].reshape(-1),
            ])
        )
        involved_users = torch.unique(batch["users"])

        return {
            "items": self.item_lookup(involved_items),
            "cates": self.cate_lookup(involved_cates),
            "user_long": self.user_long_lookup(involved_users),
            "user_short": self.user_short_lookup(involved_users),
        }

    def compute_regular_loss(
        self,
        batch,
        embed_l2=1e-6,
        layer_l2=1e-6,
        involved_embeddings=None,
    ):
        """按官方配置计算 L2，并复用辅助损失的查询结果。"""

        if involved_embeddings is None:
            involved_embeddings = self.lookup_involved_embeddings(batch)

        involved_embeddings = list(involved_embeddings.values())

        embedding_square_sum = self.item_lookup.weight.new_zeros(())

        for embedding in involved_embeddings:
            embedding_square_sum = (
                embedding_square_sum + embedding.square().sum()
            )

        # 3. 记录所有 Embedding 参数，避免被当作普通网络参数处理
        embedding_parameter_ids = set()

        for module in self.modules():
            if isinstance(module, nn.Embedding):
                for parameter in module.parameters():
                    embedding_parameter_ids.add(id(parameter))

        # 4. 对 Embedding 之外的可训练参数求平方和
        layer_square_sum = self.item_lookup.weight.new_zeros(())

        for parameter in self.parameters():
            if (
                parameter.requires_grad
                and id(parameter) not in embedding_parameter_ids
            ):
                layer_square_sum = (
                    layer_square_sum + parameter.square().sum()
                )

        # TensorFlow 的 l2_loss 是平方和的一半
        embedding_loss = 0.5 * embed_l2 * embedding_square_sum
        layer_loss = 0.5 * layer_l2 * layer_square_sum

        return embedding_loss + layer_loss


    def compute_data_loss(self, outputs, batch, num_negatives=4):
        """按正负样本分组，计算官方 softmax 推荐损失"""

        group_size = num_negatives + 1

        # 每行对应一个正样本和它的负样本
        grouped_logits = outputs["logits"].reshape(-1, group_size)
        grouped_labels = batch["labels"].reshape(-1, group_size)

        # 直接计算 log(softmax)，避免先算概率再取对数的数值问题
        log_probabilities = torch.log_softmax(
            grouped_logits,
            dim=1,
        )

        # 只保留正样本的 log 概率，负样本位置设为零
        positive_log_probabilities = torch.where(
            grouped_labels == 1,
            log_probabilities,
            torch.zeros_like(log_probabilities),
        )

        # 每组求和，再对所有组取平均
        group_losses = -positive_log_probabilities.sum(dim=1)
        data_loss = group_losses.mean()

        return data_loss


    def compute_discrepancy_loss(
        self,
        batch,
        loss_weight=0.01,
        involved_embeddings=None,
    ):
        """对去重用户计算差异损失，与正则项共享用户查询结果。"""

        if involved_embeddings is None:
            involved_users = torch.unique(batch["users"])
            involved_long_embedding = self.user_long_lookup(involved_users)
            involved_short_embedding = self.user_short_lookup(involved_users)
        else:
            involved_long_embedding = involved_embeddings["user_long"]
            involved_short_embedding = involved_embeddings["user_short"]

        squared_difference = (
            involved_long_embedding - involved_short_embedding
        ).square()

        discrepancy_loss = squared_difference.mean()

        return -loss_weight * discrepancy_loss


    def compute_contrastive_loss(
        self,
        outputs,
        batch,
        length_threshold=5,
        margin=1.0,
        loss_weight=0.1,
    ):
        """按照官方默认的 Triplet 公式计算四项对比损失"""

        long_interest = outputs["long_interest"]
        short_interest = outputs["short_interest"]
        hist_mean = outputs["hist_mean"]
        hist_recent = outputs["hist_recent"]

        # 保留每个维度的平方差，暂时不求和
        distance_long_mean = (long_interest - hist_mean).square()
        distance_long_recent = (long_interest - hist_recent).square()
        distance_short_mean = (short_interest - hist_mean).square()
        distance_short_recent = (short_interest - hist_recent).square()

        # 1. 长期兴趣应该更接近长期代理
        long_mean_recent_loss = torch.relu(
            distance_long_mean - distance_long_recent + margin
        )

        # 2. 短期兴趣应该更接近短期代理
        short_recent_mean_loss = torch.relu(
            distance_short_recent - distance_short_mean + margin
        )

        # 3. 长期代理应该更接近长期兴趣
        mean_long_short_loss = torch.relu(
            distance_long_mean - distance_short_mean + margin
        )

        # 4. 短期代理应该更接近短期兴趣
        recent_short_long_loss = torch.relu(
            distance_short_recent - distance_long_recent + margin
        )

        # 合并四项，再沿特征维度求和，得到每条样本的损失
        sample_losses = (
            long_mean_recent_loss
            + short_recent_mean_loss
            + mean_long_short_loss
            + recent_short_long_loss
        ).sum(dim=-1)

        # 只有有效历史长度严格超过阈值的样本参与计算
        sequence_length = batch["mask"].sum(dim=1)
        contrastive_mask = (
            sequence_length > length_threshold
        ).to(dtype=sample_losses.dtype)

        valid_count = contrastive_mask.sum()

        # 没有符合条件的样本时，让该批对比损失为零
        contrastive_loss = (
            sample_losses * contrastive_mask
        ).sum() / valid_count.clamp_min(1.0)

        return loss_weight * contrastive_loss

    def build_interest_proxies(self, hist_input, mask, recent_k=3):
        """对全部历史和最近历史分别求平均，得到两个兴趣代理"""

        # 全部有效历史的平均表示
        valid_mask = mask.to(dtype=hist_input.dtype)

        history_sum = (
            hist_input * valid_mask.unsqueeze(-1)
        ).sum(dim=1)

        history_count = valid_mask.sum(dim=1, keepdim=True)

        hist_mean = history_sum / history_count

        # 按官方方式，从右向左累计有效历史数量
        reversed_mask = torch.flip(valid_mask, dims=[1])
        reversed_position = torch.cumsum(reversed_mask, dim=1)
        position = torch.flip(reversed_position, dims=[1])

        # 倒数第 1 到第 k 条有效历史，属于最近历史
        recent_mask = (position >= 1) & (position <= recent_k)
        recent_mask = recent_mask.to(dtype=hist_input.dtype)

        recent_sum = (
            hist_input * recent_mask.unsqueeze(-1)
        ).sum(dim=1)

        recent_count = recent_mask.sum(dim=1, keepdim=True)

        hist_recent = recent_sum / recent_count

        return hist_mean, hist_recent

    def fuse_interests(
        self,
        fusion_state,
        target_item_embedding,
        long_interest,
        short_interest,
        time_to_now,
    ):
        """预测融合比例，组合长期兴趣和短期兴趣"""

        # 官方直接取补齐后时间张量的最后一列
        last_time_feature = time_to_now[:, -1:]

        fusion_input = torch.cat(
            [
                fusion_state,
                target_item_embedding,
                long_interest,
                short_interest,
                last_time_feature,
            ],
            dim=-1,
        )

        alpha_logit = self.fusion_mlp(fusion_input)
        alpha = torch.sigmoid(alpha_logit)

        fused_interest = (
            alpha * long_interest
            + (1.0 - alpha) * short_interest
        )

        return fused_interest, alpha

    def encode_fusion_history(self, hist_input, mask):
        """从全零状态读取历史，生成预测融合比例所需的状态"""

        batch_size = hist_input.shape[0]
        history_length = hist_input.shape[1]

        # 官方融合 GRU 的初始状态为零
        hidden = hist_input.new_zeros((batch_size, 40))

        for index in range(history_length):
            current_input = hist_input[:, index, :]

            new_hidden = self.fusion_gru(
                current_input,
                hidden,
            )

            valid_position = mask[:, index].unsqueeze(-1) == 1

            # 补零位置保持原状态
            hidden = torch.where(
                valid_position,
                new_hidden,
                hidden,
            )

        return hidden

    def encode_short_term(
        self,
        short_term_intention,
        target_item_embedding,
        rnn_outputs,
        mask,
    ):
        """计算短期注意力权重，并汇总历史输出"""

        attention_features = self.build_short_attention_features(
            short_term_intention,
            target_item_embedding,
            rnn_outputs,
        )

        batch_size = rnn_outputs.shape[0]
        history_length = rnn_outputs.shape[1]

        # 将所有历史位置展开，交给 MLP 和 BatchNorm
        flat_features = attention_features.reshape(-1, 320)

        scores = self.short_attention_mlp(flat_features)
        scores = scores.reshape(batch_size, history_length)

        # 与官方相同，用很小的分数屏蔽补零位置
        scores = scores.masked_fill(
            mask != 1,
            -(2 ** 32) + 1,
        )

        # 每条样本沿历史维度计算权重
        attention_weights = torch.softmax(scores, dim=1)

        # 对原始的 40 维 RNN 输出加权，而非映射后的 80 维特征
        weighted_history = (
            rnn_outputs * attention_weights.unsqueeze(-1)
        )

        short_interest = weighted_history.sum(dim=1)

        return short_interest, attention_weights

    def build_short_attention_features(
        self,
        short_term_intention,
        target_item_embedding,
        rnn_outputs,
    ):
        """构造短期注意力计算分数所需的特征"""

        # 40 维演化状态 + 40 维目标商品表示，得到 80 维 Query
        query = torch.cat(
            [short_term_intention, target_item_embedding],
            dim=-1,
        )

        # 每个历史输出从 40 维映射到 80 维
        attention_inputs = self.short_attention_projection(
            rnn_outputs
        )

        # 将同一条样本的 Query 放到每个历史位置
        queries = query.unsqueeze(1).expand_as(attention_inputs)

        difference = attention_inputs - queries
        product = attention_inputs * queries

        # 四组 80 维特征，拼成 320 维
        attention_features = torch.cat(
            [
                attention_inputs,
                queries,
                difference,
                product,
            ],
            dim=-1,
        )

        return attention_features

    def lookup_embeddings(self, batch):
        """根据 batch 中的索引查表，得到用户和物品向量。"""

        # 同一批用户，分别查三张用户表。
        user_embedding = self.user_lookup(batch["users"])
        user_long_embedding = self.user_long_lookup(batch["users"])
        user_short_embedding = self.user_short_lookup(batch["users"])

        # 当前候选商品：查商品表和类目表。
        item_embedding = self.item_lookup(batch["items"])
        cate_embedding = self.cate_lookup(batch["cates"])

        # 历史商品：同样查这两张表。
        item_history_embedding = self.item_lookup(batch["item_history"])
        cate_history_embedding = self.cate_lookup(
            batch["item_cate_history"]
        )

        # 候选的 32 维商品向量与 8 维类目向量，拼成 40 维。
        target_item_embedding = torch.cat(
            [item_embedding, cate_embedding], dim=-1
        )

        # 每个历史位置也拼成 40 维。
        hist_input = torch.cat(
            [item_history_embedding, cate_history_embedding], dim=-1
        )

        return {
            "user_embedding": user_embedding,
            "user_long_embedding": user_long_embedding,
            "user_short_embedding": user_short_embedding,
            "target_item_embedding": target_item_embedding,
            "hist_input": hist_input,
        }

    def build_long_attention_features(self, query, hist_input):
        """组合长期 Query 和各个历史位置，生成注意力打分特征。"""

        # 历史向量经过线性变换，形状仍为 [批大小, 历史长度, 40]。
        att_inputs = self.long_attention_projection(hist_input)

        # Query 原来是 [批大小, 40]，先增加一个历史位置维度。
        queries = query.unsqueeze(1)

        # 让同一个用户 Query 对应到每一个历史位置。
        queries = queries.expand_as(att_inputs)

        # 分别计算逐元素的差和积。
        difference = att_inputs - queries
        product = att_inputs * queries

        # 每个位置拼成 40 × 4 = 160 维特征。
        attention_features = torch.cat(
            [att_inputs, queries, difference, product],
            dim=-1,
        )

        return attention_features

    def encode_long_term(self, query, hist_input, mask):
        """计算长期注意力，并汇总出长期兴趣向量。"""

        # 复用上一步：为每个历史位置构造 160 维特征。
        attention_features = self.build_long_attention_features(
            query, hist_input
        )

        batch_size = hist_input.shape[0]
        history_length = hist_input.shape[1]

        # 暂时合并样本和历史位置，让 BatchNorm 按特征维归一化。
        flat_features = attention_features.reshape(-1, 160)

        # 每个位置输出一个分数，再恢复为 [批大小, 历史长度]。
        scores = self.long_attention_mlp(flat_features)
        scores = scores.reshape(batch_size, history_length)

        # 补齐位置设为极小的分数，使其 softmax 权重为 0。
        scores = scores.masked_fill(mask != 1, -(2 ** 32) + 1)

        # 在每条样本的历史位置之间分配权重。
        att_weights = torch.softmax(scores, dim=1)

        # 每个位置的权重，乘到这个位置的整个 40 维向量上。
        weighted_history = hist_input * att_weights.unsqueeze(-1)

        # 沿历史位置求和，得到 [批大小, 40] 的长期兴趣。
        att_fea_long = weighted_history.sum(dim=1)

        return att_fea_long, att_weights

    def encode_short_intention(self, hist_input, user_short_embedding, mask):
        """依次读取有效历史，得到短期兴趣演化状态。"""

        # 用短期用户向量作为初始记忆。
        hidden = user_short_embedding

        history_length = hist_input.shape[1]

        for index in range(history_length):
            # 取出当前历史位置的物品向量，形状为 [批大小, 40]。
            current_input = hist_input[:, index, :]

            # 根据当前输入和旧记忆，计算新记忆。
            new_hidden = self.short_evolution_gru(
                current_input, hidden
            )

            # 当前位置是否有效，形状整理为 [批大小, 1]。
            valid_position = mask[:, index].unsqueeze(-1) == 1

            # 有效位置更新记忆，补齐位置保留旧记忆。
            hidden = torch.where(
                valid_position,
                new_hidden,
                hidden,
            )

        return hidden

    def build_short_history_input(self, hist_input, batch):
        """把历史商品表示和两组时间特征拼接起来"""

        # 原来的形状是 [批次大小, 历史长度]
        # 增加最后一维，变成 [批次大小, 历史长度, 1]
        time_from_first_action = batch["time_from_first_action"].unsqueeze(-1)
        time_to_now = batch["time_to_now"].unsqueeze(-1)

        # 每个位置包含：40 维商品表示 + 两个时间数值
        short_history_input = torch.cat(
            [
                hist_input,
                time_from_first_action,
                time_to_now,
            ],
            dim=-1,
        )

        return short_history_input

    def encode_short_history(self, short_history_input, mask):
        """逐条编码历史行为，补零位置不更新状态且输出为零"""

        batch_size = short_history_input.shape[0]
        history_length = short_history_input.shape[1]
        hidden_size = self.short_history_lstm.hidden_size

        # 官方这条分支从全零状态开始
        # new_zeros 会沿用输入张量的数据类型和所在设备
        memory = short_history_input.new_zeros(
            (batch_size, hidden_size)
        )
        hidden = short_history_input.new_zeros(
            (batch_size, hidden_size)
        )

        history_outputs = []

        for index in range(history_length):
            # 取出这个位置的商品表示和两个时间特征
            current_input = short_history_input[:, index, :]

            output, new_state = self.short_history_lstm(
                current_input,
                (memory, hidden),
            )

            new_memory, new_hidden = new_state

            # [批次大小, 1]，表示各样本当前是否为有效位置
            valid_position = mask[:, index].unsqueeze(-1) == 1

            # 有效位置采用新状态，补零位置保留原状态
            memory = torch.where(
                valid_position,
                new_memory,
                memory,
            )
            hidden = torch.where(
                valid_position,
                new_hidden,
                hidden,
            )

            # 补零位置的输出为零
            current_output = torch.where(
                valid_position,
                output,
                torch.zeros_like(output),
            )

            history_outputs.append(current_output)

        # 将每一步的 [批次大小, 40] 合成完整历史输出
        rnn_outputs = torch.stack(history_outputs, dim=1)

        return rnn_outputs


class TensorFlowGRUCell(nn.Module):
    """用 PyTorch 实现 TensorFlow 1.x GRUCell 的单步计算。"""

    def __init__(self, input_size=40, hidden_size=40):
        super().__init__()

        # 输入向量和旧状态拼接后的长度。
        combined_size = input_size + hidden_size

        # 同时计算两个门，每个门都是 hidden_size 维。
        self.gates = nn.Linear(combined_size, hidden_size * 2)

        # 计算候选状态。
        self.candidate_layer = nn.Linear(combined_size, hidden_size)

        # 对齐原 GRUCell 的默认初始化。
        nn.init.xavier_uniform_(self.gates.weight)
        nn.init.ones_(self.gates.bias)

        nn.init.xavier_uniform_(self.candidate_layer.weight)
        nn.init.zeros_(self.candidate_layer.bias)


    def forward(self, inputs, hidden):
        # 当前物品向量与旧记忆拼接。
        combined = torch.cat([inputs, hidden], dim=-1)

        # sigmoid 将门的每个数限制在 0 到 1 之间。
        gates = torch.sigmoid(self.gates(combined))

        # 前一半是重置门，后一半是更新门。
        reset_gate, update_gate = gates.chunk(2, dim=-1)

        # 先用重置门筛选旧记忆，再和当前输入一起计算新内容。
        reset_hidden = reset_gate * hidden
        candidate_input = torch.cat([inputs, reset_hidden], dim=-1)
        candidate = torch.tanh(self.candidate_layer(candidate_input))

        # 更新门决定保留多少旧记忆，以及加入多少新内容。
        new_hidden = (
            update_gate * hidden
            + (1 - update_gate) * candidate
        )

        return new_hidden
