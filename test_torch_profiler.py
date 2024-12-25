#!/usr/bin/env python3
import torch
import torchvision.models as models
from torch.profiler import profile, record_function, ProfilerActivity


def memory_consumption():
    print("Memory Consumption")
    model = models.resnet18()
    inputs = torch.randn(5, 3, 224, 224)
    with profile(
        activities=[ProfilerActivity.CPU], profile_memory=True, record_shapes=True
    ) as prof:
        model(inputs)
    print(prof.key_averages().table(sort_by="self_cpu_memory_usage", row_limit=10))


def execution_time_profile():
    print("Memory Execution")
    model = models.resnet18()
    inputs = torch.randn(5, 3, 224, 224)
    print("Help: operators can call other operators. self cpu time only o")

    with profile(activities=[ProfilerActivity.CPU], record_shapes=True) as prof:
        with record_function("model_inference"):
            model(inputs)
    print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))
    print(
        prof.key_averages(group_by_input_shape=True).table(
            sort_by="cpu_time_total", row_limit=10
        )
    )


if __name__ == "__main__":
    execution_time_profile()
    memory_consumption()
