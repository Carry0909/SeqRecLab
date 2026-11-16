import torch

from data.taobao import load_vocab, iter_eval_batches
from evaluation.evaluator import evaluate_model
from models.clsr import CLSRModel


# 要测试的模型文件
CHECKPOINT_FILE = "outputs/clsr_fixed_check/best_model.pt"

# 先测试前 20 组，确认流程
# 改成 None 就是完整测试集
MAX_TEST_GROUPS = 20

TEST_NUM_NEGATIVES = 99
BATCH_SIZE = 500


def main():
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    print("测试设备 =", device)

    # 1. 读取词表
    user_vocab = load_vocab("datasets/taobao/user_vocab.pkl")
    item_vocab = load_vocab("datasets/taobao/item_vocab.pkl")
    category_vocab = load_vocab("datasets/taobao/category_vocab.pkl")

    # 2. 读取保存的检查点，先放在 CPU 上
    checkpoint = torch.load(
        CHECKPOINT_FILE,
        map_location="cpu",
        weights_only=True,
    )

    max_length = checkpoint["settings"]["max_length"]

    # 3. 创建相同结构的模型，加载训练好的参数
    model = CLSRModel(
        len(user_vocab),
        len(item_vocab),
        len(category_vocab),
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    print("模型文件 =", CHECKPOINT_FILE)
    print("保存轮次 =", checkpoint["epoch"])
    print("该模型的验证 WAUC =", checkpoint["valid_metrics"]["wauc"])

    if MAX_TEST_GROUPS is None:
        print("测试范围 = 完整测试集")
    else:
        print("测试范围 = 前", MAX_TEST_GROUPS, "个候选组")

    # 4. 顺序读取测试文件，不额外采样
    test_batches = iter_eval_batches(
        "datasets/taobao/test_data",
        user_vocab,
        item_vocab,
        category_vocab,
        batch_size=BATCH_SIZE,
        max_length=max_length,
    )

    # 5. 每组 1 个正样本 + 99 个负样本
    results = evaluate_model(
        model,
        test_batches,
        device,
        num_negatives=TEST_NUM_NEGATIVES,
        max_groups=MAX_TEST_GROUPS,
    )

    print("\n测试结果：")

    for name, value in results.items():
        print(f"{name} = {value:.4f}")


if __name__ == "__main__":
    main()