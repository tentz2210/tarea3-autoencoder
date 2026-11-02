from torch.utils.data import DataLoader, random_split
from torchvision import transforms
import pytorch_lightning as pl

from src.data.mvtec_dataset import MVTecDataset

"""
The DataModule runs the initial setup to load the MVTec image dataset
and creates the DataLoaders required for the training and testing process
"""

class MVTecDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_dir: str = "data/mvtec_ad",
        classes: list[str] | None = None,
        image_size: int = 128,
        batch_size: int = 32,
        num_workers: int = 4,
        val_split: float = 0.2,
    ):
        super().__init__()

        self.data_dir = data_dir
        self.classes = classes or ["cable", "capsule", "screw", "transistor"] #default classes
        self.image_size = image_size
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.val_split = val_split

        self.transform = transforms.Compose(
            [
                # downsamples images to 128x128
                transforms.Resize((self.image_size, self.image_size)),
                transforms.ToTensor(),
            ]
        )

    def setup(self, stage: str | None = None):
        
        # load training data split
        full_train = MVTecDataset(
            root_dir=self.data_dir,
            classes=self.classes,
            split="train",
            transform=self.transform,
        )

        val_size = int(len(full_train) * self.val_split)

        train_size = len(full_train) - val_size

        self.train_dataset, self.val_dataset = random_split(
            full_train,
            [train_size, val_size],
        )

        # load test data split
        self.test_dataset = MVTecDataset(
            root_dir=self.data_dir,
            classes=self.classes,
            split="test",
            transform=self.transform,
        )

    # create train dataloader
    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    # create validation dataloader
    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    # create test dataloader
    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
        )