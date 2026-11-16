import torch


def move_batch_to_device(batch, device):
    """将批次中的所有张量移动到指定设备"""

    device_batch = {}

    for name, tensor in batch.items():
        device_batch[name] = tensor.to(device)

    return device_batch


def clear_embedding_gradients(model):
    """清除上一批记录的原始查询梯度，避免跨批次累积。"""

    for module in model.modules():
        if isinstance(module, torch.nn.Embedding):
            if hasattr(module, "clear_gradient_parts"):
                module.clear_gradient_parts()


def clip_gradients(model, max_grad_norm=2.0):
    """Embedding 先裁剪查询贡献再合并，其他参数逐张量裁剪。"""

    embedding_parameter_ids = set()

    with torch.no_grad():
        for module in model.modules():
            if not isinstance(module, torch.nn.Embedding):
                continue
            if not hasattr(module, "gradient_parts"):
                continue

            parameter = module.weight
            embedding_parameter_ids.add(id(parameter))

            if parameter.grad is None:
                module.clear_gradient_parts()
                continue

            if not module.gradient_parts:
                raise RuntimeError("Embedding 缺少原始查询梯度，无法按 TF1 顺序裁剪")

            # 对应 TF1 IndexedSlices.values 的整体范数。
            # 不同查询的贡献先拼在一起计算范数，不提前按 ID 合并。
            squared_norm = parameter.new_zeros(())
            for indices, values in module.gradient_parts:
                squared_norm.add_(values.square().sum())

            gradient_norm = squared_norm.sqrt()
            scale = max_grad_norm / gradient_norm.clamp_min(max_grad_norm)

            # 用裁剪后的查询贡献重建稠密梯度，再交给原来的 Adam。
            parameter.grad.zero_()
            for indices, values in module.gradient_parts:
                parameter.grad.index_add_(0, indices, values * scale)

            module.clear_gradient_parts()

        # 普通参数保持官方逐参数裁剪方式。
        for parameter in model.parameters():
            if parameter.grad is None:
                continue
            if id(parameter) in embedding_parameter_ids:
                continue

            gradient = parameter.grad
            gradient_norm = torch.linalg.vector_norm(gradient)
            scale = max_grad_norm / gradient_norm.clamp_min(max_grad_norm)
            gradient.mul_(scale)


def train_one_batch(
    model,
    batch,
    optimizer,
    device,
    num_negatives=4,
    max_grad_norm=2.0,
):
    """完成一个批次的前向、反向传播和参数更新"""

    model.train()

    batch = move_batch_to_device(batch, device)

    # 清除上一个批次的梯度
    optimizer.zero_grad(set_to_none=True)
    clear_embedding_gradients(model)

    # 前向计算与总损失
    outputs = model(batch)
    losses = model.compute_loss(
        outputs,
        batch,
        num_negatives=num_negatives,
    )

    if not torch.isfinite(losses["loss"]).item():
        raise RuntimeError("总损失出现 NaN 或无穷大")

    # 反向传播、梯度裁剪、参数更新
    losses["loss"].backward()
    clip_gradients(model, max_grad_norm)
    optimizer.step()

    # 返回普通数字，供外部打印或统计
    loss_values = {}

    for name, value in losses.items():
        loss_values[name] = value.detach().item()

    return loss_values

def train_one_epoch(
    model,
    batches,
    optimizer,
    device,
    num_negatives=4,
    max_grad_norm=2.0,
    max_batches=None,
    show_step=100,
):
    """训练一轮，返回按实际样本数量加权的平均损失"""

    loss_sums = {}
    sample_count = 0
    batch_count = 0

    for batch_number, batch in enumerate(batches, start=1):
        loss_values = train_one_batch(
            model,
            batch,
            optimizer,
            device,
            num_negatives=num_negatives,
            max_grad_norm=max_grad_norm,
        )

        current_sample_count = batch["labels"].shape[0]

        for name, value in loss_values.items():
            if name not in loss_sums:
                loss_sums[name] = 0.0

            loss_sums[name] += value * current_sample_count

        sample_count += current_sample_count
        batch_count += 1

        if batch_number == 1 or batch_number % show_step == 0:
            print(
                f"  批次 {batch_number}，"
                f"当前总损失 = {loss_values['loss']:.6f}",
                flush=True,
            )

        if max_batches is not None and batch_number >= max_batches:
            break

    if sample_count == 0:
        raise RuntimeError("这一轮没有产生可训练的批次")

    average_losses = {}

    for name, value in loss_sums.items():
        average_losses[name] = value / sample_count

    return average_losses, batch_count, sample_count