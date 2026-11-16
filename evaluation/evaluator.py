import torch

from evaluation.metrics import (
    calculate_metrics,
    calculate_weighted_auc,
)
from training.steps import move_batch_to_device


@torch.no_grad()
def evaluate_model(
    model,
    batches,
    device,
    num_negatives,
    topks=(2, 4, 6),
    max_groups=None,
):
    """按顺序预测，计算排名指标和用户加权 AUC"""

    if max_groups is not None and max_groups < 1:
        raise ValueError("max_groups 必须大于零")

    group_size = num_negatives + 1
    max_samples = None

    if max_groups is not None:
        max_samples = max_groups * group_size

    all_users = []
    all_labels = []
    all_predictions = []

    was_training = model.training
    model.eval()

    try:
        for batch in batches:
            device_batch = move_batch_to_device(batch, device)
            outputs = model(device_batch)

            users = (
                device_batch["users"]
                .reshape(-1)
                .cpu()
                .tolist()
            )

            labels = (
                device_batch["labels"]
                .reshape(-1)
                .cpu()
                .tolist()
            )

            predictions = (
                outputs["predictions"]
                .reshape(-1)
                .cpu()
                .tolist()
            )

            if max_samples is not None:
                remaining = max_samples - len(all_labels)

                users = users[:remaining]
                labels = labels[:remaining]
                predictions = predictions[:remaining]

            all_users.extend(users)
            all_labels.extend(labels)
            all_predictions.extend(predictions)

            if (
                max_samples is not None
                and len(all_labels) >= max_samples
            ):
                break

    finally:
        model.train(was_training)

    results = calculate_metrics(
        all_labels,
        all_predictions,
        num_negatives=num_negatives,
        topks=topks,
    )

    results["wauc"] = calculate_weighted_auc(
        all_users,
        all_labels,
        all_predictions,
    )

    return results