import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms


def get_mnist_dataloaders(
    data_dir: str,
    batch_size: int = 128,
    num_workers: int = 4,
    pin_memory: bool = True,
):
    transform = transforms.Compose([
        transforms.Resize(32),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: 2 * x - 1.0),  # [0,1] -> [-1,1]
    ])

    train_ds = datasets.MNIST(root=data_dir, train=True, transform=transform, download=True)
    test_ds  = datasets.MNIST(root=data_dir, train=False, transform=transform, download=True)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=pin_memory)
    test_loader  = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=pin_memory)
    return train_loader, test_loader


class MNISTNoisyPairs(Dataset):
    """
    Wraps MNIST to produce (clean, noisy, label) triples for image-to-image training.

    Each item is:
        clean:  (1, 32, 32) normalised to [-1, 1]  — the generation target
        noisy:  (1, 32, 32) clean + Gaussian noise  — the condition image
        label:  int class label

    The DataLoader will therefore yield batches of shape
        x_data: (B, 1, 32, 32)
        cond:   (B, 1, 32, 32)
        y:      (B,)
    which CFGTrainer unpacks automatically when len(batch) == 3.
    """

    def __init__(self, data_dir: str, train: bool = True, noise_std: float = 0.5):
        self.noise_std = noise_std
        transform = transforms.Compose([
            transforms.Resize(32),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: 2 * x - 1.0),  # [0,1] -> [-1,1]
        ])
        self.dataset = datasets.MNIST(root=data_dir, train=train,
                                      transform=transform, download=True)

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int):
        clean, label = self.dataset[idx]                          # (1, 32, 32), int
        noisy = clean + self.noise_std * torch.randn_like(clean)  # (1, 32, 32)
        return clean, noisy, label


def get_mnist_img2img_dataloaders(
    data_dir: str,
    batch_size: int = 128,
    num_workers: int = 4,
    pin_memory: bool = True,
    noise_std: float = 0.5,
):
    """
    Returns DataLoaders yielding (clean, noisy, label) batches for
    image-to-image (denoising) flow matching.
    """
    train_ds = MNISTNoisyPairs(data_dir, train=True,  noise_std=noise_std)
    test_ds  = MNISTNoisyPairs(data_dir, train=False, noise_std=noise_std)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=pin_memory)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=pin_memory)
    return train_loader, test_loader
