import numpy as np
import glob
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
from skimage.io import imread
from torchvision.utils import save_image


def get_dataloader(batch_size, shuffle=True):

    sprite_dataset = SpritesDataset(dataset_path=DSPRITES_PATH, transform=transforms.ToTensor())

    return DataLoader(sprite_dataset, batch_size=batch_size, shuffle=shuffle)


def get_mnist(batch_size=128, path_to_data='data/mnist'):
    transform = transforms.Compose([
        transforms.Resize((32,32)),
        transforms.ToTensor()
    ])

    train_data = datasets.MNIST(path_to_data, 
                            train=True, download=True, transform=transform)
    test_data = datasets.MNIST(path_to_data, train=False, download=True, transform=transform)
    train_loader = DataLoader(train_data, batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size, shuffle=True)

    return train_loader, test_loader


def get_chairs_dataloader(batch_size=128,
                          path_to_data='data/rendered_chairs'):
    """Chair images are resized to (64, 64)."""
    all_transforms = transforms.Compose([
        transforms.Grayscale(),
        transforms.Resize((64,64)),
        transforms.ToTensor()
    ])
    chairs_data = datasets.ImageFolder(root=path_to_data,
                                       transform=all_transforms)
    chairs_loader = DataLoader(chairs_data, batch_size=batch_size,
                               shuffle=True)
    return chairs_loader

def get_chairs_test_dataloader(batch_size=62,
                               path_to_data='data/rendered_chairs_test'):
    """Each chair has 62 images"""
    all_transforms = transforms.Compose([
        transforms.Grayscale(),
        transforms.Resize((64, 64)),
        transforms.ToTensor()
    ])
    chairs_data = datasets.ImageFolder(root=path_to_data,
                                       transform=all_transforms)
    chairs_loader = DataLoader(chairs_data, batch_size=batch_size,
                               shuffle=False)
    return chairs_loader

def get_celeb(batch_size=128, 
                path='data/celebA'):
    celebA = CelebADataset(path,
                            transform=transforms.ToTensor())
    celebA_loader = DataLoader(celebA, batch_size=batch_size, shuffle=True)     
    return celebA_loader       


class CelebADataset(Dataset):

    def __init__(self, path_to_data, subsample=1, transform=None):
       
        self.img_paths = glob.glob(path_to_data + '/*')[::subsample]
        self.transform = transform

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        sample_path = self.img_paths[idx]
        sample = imread(sample_path)

        if self.transform:
            sample = self.transform(sample)
        
        return sample, 0

def main():
    
    chairs_loader = get_chairs_dataloader(path_to_data='data/rendered_chairs_small')
    for x, y in enumerate(chairs_loader):
        pass

    print(y.shape)
    
if __name__ == '__main__':
    main()
