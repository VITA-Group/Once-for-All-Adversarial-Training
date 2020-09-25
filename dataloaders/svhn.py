from torchvision import transforms
from torchvision.datasets import SVHN
from torch.utils.data import DataLoader, Subset

def svhn_dataloaders(train_batch_size=64, test_batch_size=100, num_workers=2, data_dir = 'datasets/svhn'):

    train_transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    train_set = Subset(SVHN(data_dir, split='train', transform=train_transform, download=True),list(range(68257)))
    val_set = Subset(SVHN(data_dir, split='train', transform=train_transform, download=True),list(range(68257,73257)))
    test_set = SVHN(data_dir, split='test', transform=test_transform, download=True)
            
    train_loader = DataLoader(train_set, batch_size=train_batch_size, shuffle=True, num_workers=num_workers, drop_last=True, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=test_batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=test_batch_size, shuffle=False, num_workers=2, pin_memory=True, drop_last=True)

    return train_loader, val_loader, test_loader