import torch
import dataloader

from argparse import ArgumentParser
from decorators import test_verbose
from utils import CROP_SIZE


@test_verbose
def check_loader_output(batch_size, crop_size):
    loader = dataloader.create_dataloader(
        "mnist",
        "test",
        batch_size
    )
    for pic, _ in loader:
        assert pic.size() == torch.Size([
            batch_size, 1, crop_size, crop_size
        ]), "Invalid batch dimension"
        break


parser = ArgumentParser()
parser.add_argument("--batch_size",
                    default=8,
                    required=False)

if __name__ == "__main__":
    args = parser.parse_args()
    batch_size = int(args.batch_size)
    check_loader_output(batch_size,
                        CROP_SIZE)
