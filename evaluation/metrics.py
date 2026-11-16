import math

import numpy as np
from sklearn.metrics import roc_auc_score


def calculate_metrics(
    labels,
    predictions,
    num_negatives,
    topks=(2, 4, 6),
):
    """计算全局指标和每个候选组的排名指标"""

    labels = np.asarray(labels, dtype=np.int64).reshape(-1)
    predictions = np.asarray(
        predictions,
        dtype=np.float64,
    ).reshape(-1)

    group_size = num_negatives + 1

    if len(labels) == 0:
        raise ValueError("没有可以评估的样本")

    if len(labels) != len(predictions):
        raise ValueError("标签数与预测值数量不一致")

    if len(labels) % group_size != 0:
        raise ValueError("样本数不能组成完整的候选组")

    grouped_labels = labels.reshape(-1, group_size)
    grouped_predictions = predictions.reshape(-1, group_size)

    if not np.all(grouped_labels.sum(axis=1) == 1):
        raise ValueError("每个候选组必须恰好有一个正样本")

    # 1. 全局 AUC
    results = {
        "auc": round(
            float(roc_auc_score(labels, predictions)),
            4,
        ),
    }

    # 2. 全局 LogLoss
    # 沿用官方的概率截断范围
    safe_predictions = np.clip(
        predictions,
        1e-11,
        1.0 - 1e-11,
    )

    logloss = -np.mean(
        labels * np.log(safe_predictions)
        + (1 - labels) * np.log(1 - safe_predictions)
    )

    results["logloss"] = round(float(logloss), 4)

    # 3. 准备各组指标的累计值
    mrr_sum = 0.0
    group_auc_sum = 0.0

    hit_sums = {}
    ndcg_sums = {}

    for k in topks:
        hit_sums[k] = 0.0
        ndcg_sums[k] = 0.0

    # 4. 每个候选组单独排序和计算
    for group_labels, group_predictions in zip(
        grouped_labels,
        grouped_predictions,
    ):
        # 与官方相同：argsort 后反转，得到分数降序
        order = np.argsort(group_predictions)[::-1]
        ordered_labels = group_labels[order]

        # 数组下标从 0 开始，排名从 1 开始
        positive_rank = int(
            np.where(ordered_labels == 1)[0][0]
        ) + 1

        mrr_sum += 1.0 / positive_rank

        for k in topks:
            if positive_rank <= k:
                hit_sums[k] += 1.0

                # 每组只有一个正样本，理想 DCG 为 1
                ndcg_sums[k] += (
                    1.0 / math.log2(positive_rank + 1)
                )

        # 单正样本组的 AUC：
        # 正样本胜过一个负样本计 1 分，同分计 0.5 分
        positive_score = group_predictions[
            group_labels == 1
        ][0]

        negative_scores = group_predictions[
            group_labels == 0
        ]

        wins = np.sum(positive_score > negative_scores)
        ties = np.sum(positive_score == negative_scores)

        group_auc = (
            wins + 0.5 * ties
        ) / len(negative_scores)

        group_auc_sum += float(group_auc)

    # 5. 按候选组数量取平均
    group_count = len(grouped_labels)

    results["mean_mrr"] = round(mrr_sum / group_count, 4)

    for k in topks:
        results[f"hit@{k}"] = round(
            hit_sums[k] / group_count,
            4,
        )
        results[f"ndcg@{k}"] = round(
            ndcg_sums[k] / group_count,
            4,
        )

    results["group_auc"] = round(
        group_auc_sum / group_count,
        4,
    )

    return results

def calculate_weighted_auc(users, labels, predictions):
    """按照官方定义，按用户样本数量加权平均用户 AUC"""

    users = np.asarray(users, dtype=np.int64).reshape(-1)
    labels = np.asarray(labels, dtype=np.int64).reshape(-1)
    predictions = np.asarray(
        predictions,
        dtype=np.float64,
    ).reshape(-1)

    if len(users) == 0:
        raise ValueError("没有可以计算 WAUC 的样本")

    if not (
        len(users) == len(labels) == len(predictions)
    ):
        raise ValueError("用户、标签和预测值的数量不一致")

    # 按用户排序，让同一个用户的样本相邻
    order = np.argsort(users)

    sorted_users = users[order]
    sorted_labels = labels[order]
    sorted_predictions = predictions[order]

    # 找到每个用户在排序后数组中的起点和样本数量
    _, user_starts, user_counts = np.unique(
        sorted_users,
        return_index=True,
        return_counts=True,
    )

    weighted_auc_sum = 0.0

    for start, count in zip(user_starts, user_counts):
        end = start + count

        user_labels = sorted_labels[start:end]
        user_predictions = sorted_predictions[start:end]

        if np.unique(user_labels).size != 2:
            raise ValueError(
                "计算 WAUC 时，每位用户都需要同时有正负样本"
            )

        user_auc = roc_auc_score(
            user_labels,
            user_predictions,
        )

        weighted_auc_sum += float(user_auc) * int(count)

    wauc = weighted_auc_sum / len(users)

    # 与官方一致，最终保留四位小数
    return round(wauc, 4)