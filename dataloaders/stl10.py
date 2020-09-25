from torchvision import transforms
from torchvision.datasets import STL10
from torch.utils.data import DataLoader, Subset

def stl10_dataloaders(train_batch_size=64, test_batch_size=100, num_workers=2, data_dir = 'datasets/stl10'):

    train_transform = transforms.Compose([
                    transforms.Pad(4),
                    transforms.RandomCrop(96),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    train_set = STL10(data_dir, split='train', download=True, transform=train_transform)
    test_set = STL10(data_dir, split='test', download=True, transform=test_transform)

    train_loader = DataLoader(train_set, batch_size=train_batch_size, shuffle=True, num_workers=num_workers,
                                drop_last=True, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=test_batch_size, shuffle=False, num_workers=2, pin_memory=True)

    return train_loader, test_loader
