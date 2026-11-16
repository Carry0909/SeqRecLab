import json
import math
import os
import random
import time

import numpy as np
import torch

from data.taobao import (
    load_vocab,
    get_line_offsets,
    iter_train_batches,
    iter_eval_batches,
)
from evaluation.evaluator import evaluate_model
from models.clsr import CLSRModel
from training.optimizers import TensorFlowAdam
from training.steps import train_one_epoch


# ==================================================
# 配置
# ==================================================

# 先小规模跑通整个流程，再改为 False
SMOKE_TEST = False

SEED = 42
MAX_LENGTH = 50

TRAIN_NUM_NEGATIVES = 4
VALID_NUM_NEGATIVES = 4
TEST_NUM_NEGATIVES = 99

LEARNING_RATE = 0.001
MAX_GRAD_NORM = 2.0

# 按官方启动脚本，使用 WAUC 选最佳模型
SELECTION_METRIC = "wauc"
PATIENCE = 5

EVAL_BATCH_SIZE = 500

if SMOKE_TEST:
    EPOCHS = 2
    TRAIN_BATCH_SIZE = 8

    MAX_TRAIN_BATCHES = 3
    MAX_VALID_GROUPS = 20
    MAX_TEST_GROUPS = 20

    OUTPUT_DIR = "outputs/clsr_smoke"
else:
    # 官方启动脚本默认配置
    EPOCHS = 100
    TRAIN_BATCH_SIZE = 500

    MAX_TRAIN_BATCHES = None
    MAX_VALID_GROUPS = None
    MAX_TEST_GROUPS = None

    OUTPUT_DIR = "outputs/clsr_fixed_full"


def main():
    # ==================================================
    # 1. 初始化
    # ==================================================

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("运行模式 =", "小规模流程检查" if SMOKE_TEST else "正式训练")
    print("使用设备 =", device)
    print("输出目录 =", OUTPUT_DIR)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    checkpoint_file = os.path.join(
        OUTPUT_DIR,
        "best_model.pt",
    )

    train_file = "datasets/taobao/train_data"
    valid_file = "datasets/taobao/valid_data"
    test_file = "datasets/taobao/test_data"

    # ==================================================
    # 2. 读取词表、创建模型和优化器
    # ==================================================

    user_vocab = load_vocab("datasets/taobao/user_vocab.pkl")
    item_vocab = load_vocab("datasets/taobao/item_vocab.pkl")
    category_vocab = load_vocab("datasets/taobao/category_vocab.pkl")

    model = CLSRModel(
        len(user_vocab),
        len(item_vocab),
        len(category_vocab),
    ).to(device)

    optimizer = TensorFlowAdam(
        model.parameters(),
        learning_rate=LEARNING_RATE,
    )

    print("\n正在建立训练文件索引……", flush=True)
    line_offsets = get_line_offsets(train_file)
    print("训练文件样本数 =", len(line_offsets))

    # 从负无穷开始，保证第一轮即使 WAUC 为零也会保存
    best_metric = -float("inf")
    best_epoch = 0

    # ==================================================
    # 3. 逐轮训练与验证
    # ==================================================

    for epoch in range(1, EPOCHS + 1):
        start_time = time.perf_counter()

        print(f"\n========== 第 {epoch} 轮 ==========", flush=True)

        # 每轮重新创建迭代器，内部重新打乱样本位置
        train_batches = iter_train_batches(
            train_file,
            line_offsets,
            user_vocab,
            item_vocab,
            category_vocab,
            batch_size=TRAIN_BATCH_SIZE,
            max_length=MAX_LENGTH,
            num_negatives=TRAIN_NUM_NEGATIVES,
        )

        average_losses, batch_count, sample_count = train_one_epoch(
            model,
            train_batches,
            optimizer,
            device,
            num_negatives=TRAIN_NUM_NEGATIVES,
            max_grad_norm=MAX_GRAD_NORM,
            max_batches=MAX_TRAIN_BATCHES,
        )

        print("本轮训练批次数 =", batch_count)
        print("本轮训练样本数（含负样本）=", sample_count)

        print("平均训练损失：")
        for name, value in average_losses.items():
            print(f"  {name} = {value:.6f}")

        # 每轮重新创建验证迭代器，从文件开头读取
        valid_batches = iter_eval_batches(
            valid_file,
            user_vocab,
            item_vocab,
            category_vocab,
            batch_size=EVAL_BATCH_SIZE,
            max_length=MAX_LENGTH,
        )

        print("正在验证……", flush=True)

        valid_results = evaluate_model(
            model,
            valid_batches,
            device,
            num_negatives=VALID_NUM_NEGATIVES,
            max_groups=MAX_VALID_GROUPS,
        )

        print("验证结果：")
        for name, value in valid_results.items():
            print(f"  {name} = {value:.4f}")

        current_metric = valid_results[SELECTION_METRIC]

        if not math.isfinite(current_metric):
            raise RuntimeError("选模指标出现 NaN 或无穷大")

        # 严格提升才替换最佳模型，相同分数不算提升
        if current_metric > best_metric:
            best_metric = current_metric
            best_epoch = epoch

            checkpoint = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "valid_metrics": valid_results,
                "settings": {
                    "seed": SEED,
                    "max_length": MAX_LENGTH,
                    "train_num_negatives": TRAIN_NUM_NEGATIVES,
                    "valid_num_negatives": VALID_NUM_NEGATIVES,
                    "test_num_negatives": TEST_NUM_NEGATIVES,
                    "train_batch_size": TRAIN_BATCH_SIZE,
                    "learning_rate": LEARNING_RATE,
                    "smoke_test": SMOKE_TEST,
                },
            }

            torch.save(checkpoint, checkpoint_file)

            print(
                f"已保存最佳模型：第 {best_epoch} 轮，"
                f"{SELECTION_METRIC} = {best_metric:.4f}"
            )
        else:
            print(
                f"本轮未提升，距离最佳轮次已有 "
                f"{epoch - best_epoch} 轮"
            )

        elapsed = time.perf_counter() - start_time
        print(f"本轮耗时 = {elapsed:.1f} 秒")

        if PATIENCE > 0 and epoch - best_epoch >= PATIENCE:
            print("达到早停条件，结束训练")
            break

    # ==================================================
    # 4. 加载最佳模型，再评估测试集
    # ==================================================

    print(f"\n加载最佳模型：{checkpoint_file}", flush=True)
    print(f"\n加载最佳模型：{checkpoint_file}", flush=True)

    checkpoint = torch.load(
        checkpoint_file,
        map_location=device,
        weights_only=True,
    )

    model.load_state_dict(checkpoint["model_state_dict"])

    print("最佳模型来自第", checkpoint["epoch"], "轮")

    test_batches = iter_eval_batches(
        test_file,
        user_vocab,
        item_vocab,
        category_vocab,
        batch_size=EVAL_BATCH_SIZE,
        max_length=MAX_LENGTH,
    )

    print("正在测试……", flush=True)

    test_results = evaluate_model(
        model,
        test_batches,
        device,
        num_negatives=TEST_NUM_NEGATIVES,
        max_groups=MAX_TEST_GROUPS,
    )

    print("\n最佳模型的测试结果：")
    for name, value in test_results.items():
        print(f"  {name} = {value:.4f}")

    # ==================================================
    # 5. 保存评估结果
    # ==================================================

    result_file = os.path.join(OUTPUT_DIR, "results.json")

    results = {
        "smoke_test": SMOKE_TEST,
        "best_epoch": checkpoint["epoch"],
        "best_valid_metrics": checkpoint["valid_metrics"],
        "test_metrics": test_results,
    }

    with open(result_file, "w", encoding="utf-8") as file:
        json.dump(
            results,
            file,
            ensure_ascii=False,
            indent=2,
        )

    print("\n模型已保存到 =", checkpoint_file)
    print("结果已保存到 =", result_file)


if __name__ == "__main__":
    main()