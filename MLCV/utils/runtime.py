import torch


def get_device(prefer_cuda: bool = True) -> torch.device:
    if prefer_cuda and torch.cuda.is_available():
        print("All good, a Gpu is available.")
        return torch.device("cuda:0")

    print("Please set GPU via Edit -> Notebook Settings.")
    return torch.device("cpu")
