import math

import torch


class TensorFlowAdam(torch.optim.Optimizer):
    """按 TensorFlow 1.x Adam 的公式更新稠密梯度参数"""

    def __init__(
        self,
        parameters,
        learning_rate=0.001,
        beta1=0.9,
        beta2=0.999,
        epsilon=1e-8,
    ):
        defaults = {
            "lr": learning_rate,
            "beta1": beta1,
            "beta2": beta2,
            "epsilon": epsilon,
            "step_count": 0,
        }

        super().__init__(parameters, defaults)

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            group["step_count"] += 1
            step_count = group["step_count"]

            beta1 = group["beta1"]
            beta2 = group["beta2"]
            epsilon = group["epsilon"]

            # 官方 Adam 的偏差修正放在学习率上
            corrected_lr = (
                group["lr"]
                * math.sqrt(1.0 - beta2 ** step_count)
                / (1.0 - beta1 ** step_count)
            )

            for parameter in group["params"]:
                if parameter.grad is None:
                    continue

                gradient = parameter.grad
                state = self.state[parameter]

                # 第一次更新时，为这个参数创建两组历史统计量
                if len(state) == 0:
                    state["first_moment"] = torch.zeros_like(parameter)
                    state["second_moment"] = torch.zeros_like(parameter)

                first_moment = state["first_moment"]
                second_moment = state["second_moment"]

                # 梯度的一阶、二阶移动平均
                first_moment.mul_(beta1)
                first_moment.add_(gradient, alpha=1.0 - beta1)

                second_moment.mul_(beta2)
                second_moment.addcmul_(
                    gradient,
                    gradient,
                    value=1.0 - beta2,
                )

                denominator = second_moment.sqrt() + epsilon

                parameter.addcdiv_(
                    first_moment,
                    denominator,
                    value=-corrected_lr,
                )