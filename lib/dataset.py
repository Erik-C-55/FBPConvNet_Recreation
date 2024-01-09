"""dataset.py

Creates the dataset, dataloader for use in training/evaluating the model."""

# Standard Imports
import os
import typing

# Third-Party Imports
import torch
import numpy as np
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import Compose, RandomHorizontalFlip, \
    RandomVerticalFlip


class FBPDataset(Dataset):
    """A class for the FBPConvNet Datataset
    
    Parameters
    ----------
    n_ellipse : (int, int)
        (min, max) # of ellipses per image. Helps find pre-generated image dir
    full_views : int
        Number of views used for "full-view" filtered back projection
    low_views : int
        Reduced number of views used for low-view filtered back projection
    transform : torch.nn.Module
        Any torchvision transforms to be applied to this data
    mode : str
        'train', 'validation', or 'test'
    n_samps: int
        Number of samples to use for all the data (train+val+test)
    """
    def __init__(self, n_ellipse: typing.Tuple[int]=(25, 34), 
                 full_views: int=1000, low_views: int=50,
                 transform: torch.nn.Module=None, mode: str='train',
                 n_samps: int=500):
        self.n_ellipse = n_ellipse
        self.im_size = 512
        self.full_views = full_views
        self.low_views = low_views
        self.transform = transform
        self.full_views_list = []
        self.low_views_list = []
        
        self.obtainImList(n_ellipse, n_samps, mode)
        
    def obtainImList(self, n_ellipse: int, n_samps: int, mode: str):
        """Obtains a list of filepaths for dataset images"""
        
        # Name of directory where images are stored
        dirName = os.path.join('imageData',
                               f'{n_ellipse[0]}_{n_ellipse[1]}_Images')
        
        # Since the images are randomly generated anyway, the training set can
        # simply be the first 90% of images, with the next 5% for validation &
        # the last 5% for test
        train_Ims = int(0.9*n_samps)
        val_Ims = int(np.floor(0.05*n_samps))
        test_Ims = int(np.ceil(0.05*n_samps))
            
        # Calculate the ranges for each set
        if mode == 'train':
            lower = 1
            upper = train_Ims + 1
            
        elif mode == 'validation':
            lower = train_Ims + 1
            upper = train_Ims + val_Ims + 1
            
        elif mode == 'test':
            lower = train_Ims + val_Ims + 1
            upper = train_Ims + val_Ims + test_Ims + 1
            
        for im in range(lower, upper):
            imName = 'im_' + str(im).zfill(4) + '.npy'
            self.full_views_list.append(os.path.join(
                dirName, f'{self.full_views}_Views', imName))
            self.low_views_list.append(os.path.join(
                dirName, f'{self.low_views}_Views', imName))
        
    def __getitem__(self, index):
        
        # Load ground truth and input as tensors
        full_fbp = torch.from_numpy(np.load(self.full_views_list[index]))
        low_fbp = torch.from_numpy(np.load(self.low_views_list[index]))
       
        # Expand tensors to preserve channel dimension.
        full_fbp = full_fbp.unsqueeze(dim=0)
        low_fbp = low_fbp.unsqueeze(dim=0)
        
        # Transform ground truth and input. To ensure ground truth and low-view
        # images receive the same transformation, first combine them along a 
        # 'frames' dimension to give a tensor of dimensions
        # [Frames =2, Channels=1, Height, Width]
        
        if self.transform is not None:
            transformed = self.transform(torch.stack((full_fbp, low_fbp),
                                                     dim=0))
            full_fbp = transformed[0,:,:,:]
            low_fbp = transformed[1,:,:,:]
        
        return low_fbp, full_fbp
    
    def __len__(self) -> int:
        """Returns dataset length"""
        return len(self.full_views_list)
    
        
if __name__ == '__main__':
        
    transform = Compose([RandomHorizontalFlip(), RandomVerticalFlip()])
    
    imLoader = DataLoader(FBPDataset(n_ellipse=(25,34), transform=transform),
                          batch_size=2, shuffle=False, num_workers=1)
    
    low_fbp, full_fbp = iter(imLoader).next()
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(8, 8))
    
    ax1.imshow(low_fbp[0,:,:,:].squeeze(0), cmap=plt.cm.Greys_r)
    ax1.set_title('50-view FBP 1')
    
    ax2.imshow(low_fbp[1,:,:,:].squeeze(0), cmap=plt.cm.Greys_r)
    ax2.set_title('50-view FBP 2')
    
    ax3.imshow(full_fbp[0,:,:,:].squeeze(0), cmap=plt.cm.Greys_r)
    ax3.set_title('1000-view FBP 1')
    
    ax4.imshow(full_fbp[1,:,:,:].squeeze(0), cmap=plt.cm.Greys_r)
    ax4.set_title('1000-view FBP 2')
    
    fig.tight_layout()
    plt.show()
    
