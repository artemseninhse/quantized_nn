from torch.utils.data import Dataset, DataLoader
from utils import (
    DATASETS,
    NUM_WORKERS,
    TRANSFORMS
)


def create_dataloader(dataset_name,
                      transform,
                      batch_size,
                      ):
    dataset = load_dataset(dataset_name, transform)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=NUM_WORKERS
    )
    return loader


def load_dataset(dataset_name, transform):
    return DATASETS[dataset_name](
        "./data",
        download=True,
        transform=TRANSFORMS[transform]
    )

