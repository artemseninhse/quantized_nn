import dataloader

from argparse import ArgumentParser
from utils import SEED, set_seed

parser = ArgumentParser()
parser.add_argument("--dataset_name", required=True)
parser.add_argument("--test",
                    default=False,
                    action="store_true",
                    required=False)
parser.add_argument("--batch_size",
                    default=8,
                    required=False)

if __name__ == "__main__":
    args = parser.parse_args()
    dataset_name = args.dataset_name
    batch_size = int(args.batch_size)
    transform = "train"
    if args.test:
        transform = "test"
    set_seed(SEED)
    loader = dataloader.create_dataloader(
        dataset_name,
        transform,
        batch_size
    )
    for pic, lab in loader:
        print(pic.size(), lab.size())
        break
    print("="*50)
    print(f"Test of {dataset_name} is finished")

