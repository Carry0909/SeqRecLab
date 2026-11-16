import pickle
import math
import random
import torch


def load_vocab(file_path):
    """读取官方词表文件，返回 ID 到索引的映射字典"""

    # pkl 是二进制文件，用"rb"打开
    with open(file_path, "rb") as f:
        vocabulary= pickle.load(f)

    return vocabulary


def parse_line(line, user_vocab, item_vocab, category_vocab):
    """解析一行样本，转换 ID ，并保留历史时间"""

    # 去掉行尾换行符，按照 \t 拆成 8 列
    columns = line.rstrip("\r\n").split("\t")

    label = int(columns[0])

    user_id = user_vocab.get(columns[1],0)
    item_id = item_vocab.get(columns[2],0)
    category_id = category_vocab.get(columns[3],0)

    timestamp = float(columns[4])

    history_items = []
    for original_item_id in columns[5].split(","):
        item_index = item_vocab.get(original_item_id,0)
        history_items.append(item_index)

    history_categories = []
    for original_category_id in columns[6].split(","):
        category_index = category_vocab.get(original_category_id,0)
        history_categories.append(category_index)

    history_timestamps = []
    for original_timestamp in columns[7].split(","):
        history_timestamps.append(float(original_timestamp))

    sample = {
        "label": label,
        "user_id": user_id,
        "item_id": item_id,
        "category_id": category_id,
        "timestamp": timestamp,
        "history_items": history_items,
        "history_categories": history_categories,
        "history_timestamps": history_timestamps,
    }

    return sample


def calculate_time_features(timestamp, history_timestamps):
    """按官方 Taobao 读取器的规则，计算三组时间特征"""

    # 源码缩放系数
    time_scale = 3600 * 24 /1000

    # 把目标时间接到历史时间后面
    all_timestamps = history_timestamps.copy()
    all_timestamps.append(timestamp)

    first_time = history_timestamps[0]

    time_diff = [] # 相邻交互的时间差
    time_from_first_action= [] # 从第一次行为起经过的时间
    time_to_now = [] # 行为历史距离当前目标的时间

    for index in range(len(history_timestamps)):
        history_time = history_timestamps[index]
        next_time = all_timestamps[index+1]

        # 当前历史到下一次交互的间隔
        interval = (next_time-history_time)/time_scale
        interval = max(interval,0.5)
        time_diff.append(math.log(interval))

        # 首条历史到下一次交互的间隔
        elapsed = (next_time-first_time)/time_scale
        elapsed = max(elapsed,0.5)
        time_from_first_action.append(math.log(elapsed))

        # 当前历史距离目标交互的时间
        age = (timestamp-history_time)/time_scale
        age = max(age,0.5)
        time_to_now.append(math.log(age))

    return {
        "time_diff": time_diff,
        "time_from_first_action": time_from_first_action,
        "time_to_now": time_to_now,
    }


def pad_sequence(sequence, max_length=50):
    """保留最近的历史，在右侧补零，并生成 mask"""

    sequence = sequence[-max_length:]

    length = len(sequence)

    mask = [1] * length # 有效历史对应位置为1

    padding_length = max_length - length
    sequence = sequence + [0] * padding_length
    mask = mask + [0] * padding_length

    return sequence, mask, length


def prepare_sample(line, user_vocab, item_vocab, category_vocab, max_length=50):
    """解析一行样本，计算时间特征，再统一截断和补齐"""
    sample = parse_line(line, user_vocab, item_vocab, category_vocab)

    # noinspection bad-argument-type
    time_features = calculate_time_features(
        sample["timestamp"],
        sample["history_timestamps"],
    )

    history_items, mask, length = pad_sequence(
        sample["history_items"],max_length
    )
    sample["history_items"] = history_items
    sample["history_mask"] = mask
    sample["history_length"] = length

    history_categories, _, _ = pad_sequence(
        sample["history_categories"],max_length
    )
    sample["history_categories"] = history_categories

    for name, values in time_features.items():
        padded_values, _, _ = pad_sequence(values, max_length)
        sample[name] = padded_values

    return sample


def build_batch(samples):
    """把多条处理好的样本，按字段合并为 pytorch 张量"""

    # ID 和长度使用整数类型，时间特征 / 标签 / mask 使用浮点类型
    field_types = {
        "label": torch.float32,
        "user_id": torch.long,
        "item_id": torch.long,
        "category_id": torch.long,
        "timestamp": torch.float32,
        "history_items": torch.long,
        "history_categories": torch.long,
        "history_mask": torch.float32,
        "history_length": torch.long,
        "time_diff": torch.float32,
        "time_from_first_action": torch.float32,
        "time_to_now": torch.float32,
    }

    batch = {}

    for name, data_type in field_types.items():
        # 收集所有样本中，同一个字段的值
        values = []
        for sample in samples:
            values.append(sample[name])

        # 将列表转为张量
        batch[name] = torch.tensor(values, dtype=data_type)

    # 标签整理成 [batch_size, 1]。
    batch["label"] = batch["label"].reshape(-1, 1)

    # 计算每条样本中，历史类目与候选类目相同的比例。
    attn_labels = []

    for sample in samples:
        match_count = 0
        history_length = sample["history_length"]
        target_category = sample["category_id"]

        # 只遍历有效历史，跳过右侧补齐的位置。
        for index in range(history_length):
            history_category = sample["history_categories"][index]

            if history_category == target_category:
                match_count += 1

        ratio = match_count / history_length
        attn_labels.append(ratio)

    # 每条样本一个比例，整理成 [batch_size, 1]。
    batch["attn_labels"] = torch.tensor(
        attn_labels, dtype=torch.float32
    ).reshape(-1, 1)

    # 在输出时，将我们的单样本字段名对应到官方 batch 字段名。
    return {
        "labels": batch["label"],
        "users": batch["user_id"],
        "items": batch["item_id"],
        "cates": batch["category_id"],
        "time": batch["timestamp"],
        "item_history": batch["history_items"],
        "item_cate_history": batch["history_categories"],
        "mask": batch["history_mask"],
        "time_diff": batch["time_diff"],
        "time_from_first_action": batch["time_from_first_action"],
        "time_to_now": batch["time_to_now"],
        "attn_labels": batch["attn_labels"],
    }

def add_training_negatives(samples, num_negatives=4):
    """为每条正样本添加 batch 内采样的负例，无法采样时跳过该批。"""

    # 官方训练读取器会跳过少于 5 条正样本的 batch。
    if len(samples) < 5:
        return []

    # 收集当前 batch 的目标商品。
    item_ids = []
    for sample in samples:
        item_ids.append(sample["item_id"])

    # 所有目标商品都相同，无法抽到不同的商品，跳过该批。
    if len(set(item_ids)) < 2:
        return []

    training_samples = []

    for sample in samples:
        # 每组先放入一条正样本。
        positive_sample = sample.copy()
        positive_sample["label"] = 1
        training_samples.append(positive_sample)

        negative_count = 0

        while negative_count < num_negatives:
            # 从原始 batch 中随机抽取一个候选商品。
            sampled_index = random.randint(0, len(samples) - 1)
            sampled_sample = samples[sampled_index]

            # 与正目标相同就重新抽。
            if sampled_sample["item_id"] == sample["item_id"]:
                continue

            # 保留当前用户及历史，换成采样商品和对应类目。
            negative_sample = sample.copy()
            negative_sample["label"] = 0
            negative_sample["item_id"] = sampled_sample["item_id"]
            negative_sample["category_id"] = sampled_sample["category_id"]

            training_samples.append(negative_sample)
            negative_count += 1

    return training_samples

def iter_eval_batches(
    file_path,
    user_vocab,
    item_vocab,
    category_vocab,
    batch_size=500,
    max_length=50,
):
    """按文件顺序读取验证或测试样本，逐批返回张量。"""

    samples = []

    with open(file_path, "r", encoding="utf-8") as file:
        for line in file:
            # 跳过空行。
            if not line.strip():
                continue

            # 准备一条样本，放入当前批次。
            sample = prepare_sample(
                line,
                user_vocab,
                item_vocab,
                category_vocab,
                max_length,
            )
            samples.append(sample)

            # 收集够一个 batch，就交给调用方。
            if len(samples) == batch_size:
                yield build_batch(samples)
                samples = []

    # 文件末尾剩余的样本，也组成一个 batch。
    if samples:
        yield build_batch(samples)

def get_line_offsets(file_path):
    """记录每条非空样本在文件中的起始位置。"""

    offsets = [] # 结果列表，存放每行的起始字节偏移

    with open(file_path, "rb") as file:
        while True:
            # 先记住当前行开始的位置，再读取这一行。
            offset = file.tell()
            line = file.readline()

            # 没有内容，说明已经读到文件末尾。
            if line == b"":
                break

            # 非空行才是一条样本，保存它的位置。
            if line.strip():
                offsets.append(offset)

    return offsets


def iter_train_batches(
    file_path,
    offsets,
    user_vocab,
    item_vocab,
    category_vocab,
    batch_size=500,
    max_length=50,
    num_negatives=4,
):
    """随机读取训练样本，添加负例，逐批返回张量。"""

    # 复制后打乱，保留原始位置列表，方便下个 epoch 复用。
    shuffled_offsets = offsets.copy()
    random.shuffle(shuffled_offsets)

    samples = []

    with open(file_path, "rb") as file:
        for offset in shuffled_offsets:
            # 跳到这条样本的起点，读出一行，再转换为文字。
            file.seek(offset)
            line = file.readline().decode("utf-8")

            sample = prepare_sample(
                line,
                user_vocab,
                item_vocab,
                category_vocab,
                max_length,
            )
            samples.append(sample)

            # 凑够一批正样本后，进行负采样并组批。
            if len(samples) == batch_size:
                training_samples = add_training_negatives(
                    samples, num_negatives
                )

                if training_samples:
                    yield build_batch(training_samples)

                samples = []

    # 处理文件末尾剩余的正样本。
    if samples:
        training_samples = add_training_negatives(
            samples, num_negatives
        )

        if training_samples:
            yield build_batch(training_samples)