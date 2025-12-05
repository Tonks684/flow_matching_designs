from torchvision import datasets, transforms
from torch.utils.data import DataLoader

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
