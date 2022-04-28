import os
import torch
import numpy as np
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader, Dataset

class FBPDataset(Dataset):
    
    def __init__(self, n_ellipse=(25,34), full_views=1000, low_views=143,
                 transform=None, mode='train', n_samps=500):
        self.n_ellipse = n_ellipse
        self.im_size = 512
        self.full_views = full_views
        self.low_views = low_views
        self.transform = transform
        self.full_views_list = []
        self.low_views_list = []
        
        self.obtainImList(n_ellipse, n_samps, mode)
        
    def obtainImList(self, n_ellipse, n_samps, mode):
        
        # Name of directory where images are stored
        dirName = os.path.join('imageData',str(n_ellipse[0]) + '_' + str(n_ellipse[1]) +'_Images')
        
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
            self.full_views_list.append(os.path.join(dirName, str(self.full_views) + '_Views', imName))
            self.low_views_list.append(os.path.join(dirName, str(self.low_views) + '_Views', imName))
        
    def __getitem__(self, index):
        
        # Load ground truth and input as tensors
        full_fbp = torch.from_numpy(np.load(self.full_views_list[index]))
        low_fbp = torch.from_numpy(np.load(self.low_views_list[index]))
       
        # Transform ground truth and input
        if self.transform is not None:
            full_fbp = self.transform(full_fbp)
            low_fbp = self.transform(low_fbp)
        
        # Expand tensors to preserve channel dimension
        full_fbp = full_fbp.unsqueeze(dim=0)
        low_fbp = low_fbp.unsqueeze(dim=0)
        
        return low_fbp, full_fbp
    
    def __len__(self):
        return len(self.full_views_list)
       
if __name__ == '__main__':
    imLoader = DataLoader(FBPDataset(n_ellipse=(25,34)), batch_size=4,
                         shuffle=False, num_workers=4)
    
    low_fbp, full_fbp = iter(imLoader).next()
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4.5))
    
    ax1.imshow(low_fbp, cmap=plt.cm.Greys_r, vmin=0, vmax=1)
    ax1.title('Low-view FBP')
    
    ax2.imshow(full_fbp, cmap=plt.cm.Greys_r, vmin=0, vmax=1)
    ax2.title('Full-view FBP')
    
    fig.tight_layout()
    plt.show()
    
