import os
import torch
import numpy as np
import matplotlib.pyplot as plt

from torch_radon import ParallelBeam
from torch.utils.data import DataLoader, Dataset

class FBPDataset(Dataset):
    
    def __init__(self, n_ellipse=(25,34), full_views=1000, low_views=143,
                 transform=None, mode='train', n_samps=500):
        self.n_ellipse = n_ellipse
        self.im_size = 512
        self.full_views = full_views
        self.low_views = low_views
        self.transform = transform
        self.fnames = []
        
        self.obtainImList(n_ellipse, n_samps, mode)
        
    def obtainImList(self, n_ellipse, n_samps, mode):
        
        # Name of directory where images are stored
        dirName = 'imageData/' + str(n_ellipse[0]) + '_' + str(n_ellipse[1]) +'_Images'
        
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
            self.fnames.append(os.path.join(dirName, 'im_' + \
                                            str(im).zfill(4) + '.npy'))
        
    def __getitem__(self, index):
        origIm = np.load(self.fnames[index])
        
        if self.transform is not None:
            origIm = self.transform(origIm)
        
        # Calculate all view angles
        angles = np.linspace(0, np.pi, self.full_views, endpoint=False)

        # Calculate detector count (default from torch_radon example)
        det_count = int(np.sqrt(2)*self.im_size + 0.5)
        
        # # Move images to GPU for speed
        # if torch.cuda.is_available():
        #     device = torch.device('cuda')
            
        origIm = torch.FloatTensor(origIm).to(self.device)
            
        # Generate Radon transform for full-view parallel-beam CT
        radon = ParallelBeam(det_count=det_count, angles=angles)
        
        # Calculate full-view sinogram and filtered backprojection
        full_sgram = radon.forward(origIm)
        full_fbp = radon.backprojection(radon.filter_sinogram(full_sgram,
                                                             filter_name='ramp'))
        
        # Calculate all view angles
        angles = np.linspace(0, np.pi, self.low_views, endpoint=False)
        
        # Generate Radon transform for low-view parallel-beam CT
        radon = ParallelBeam(det_count=det_count, angles=angles)
        
        # Calculate low-view sinogram and filtered backprojection
        low_sgram = radon.forward(origIm)
        low_fbp = radon.backprojection(radon.filter_sinogram(low_sgram,
                                                             filter_name='ramp'))
        
        return low_fbp, full_fbp, low_sgram, full_sgram
    
    def __len__(self):
        return len(self.fnames)
       
if __name__ == '__main__':
    imLoader = DataLoader(FBPDataset('imageData/5_14_Ellipses'), batch_size=1,
                         shuffle=False, num_workers=0)
    
    low_fbp, full_fbp, low_sgram, full_sgram = iter(imLoader).next()
    
    plt.imshow(low_fbp, cmap='gray', vmin=0, vmax=1)
    plt.suptitle('Low-view FBP')
    plt.title('Close to continue.')     
    plt.show()
    
    plt.imshow(full_fbp, cmap='gray', vmin=0, vmax=1)
    plt.suptitle('Full-view FBP')
    plt.title('Close to continue.')     
    plt.show()
    
    plt.imshow(low_sgram, cmap='gray', vmin=0, vmax=1)
    plt.suptitle('Low-view Sinogram')
    plt.title('Close to continue.')     
    plt.show()
    
    plt.imshow(full_sgram, cmap='gray', vmin=0, vmax=1)
    plt.suptitle('Full-view sinogram')
    plt.title('Close to continue.')     
    plt.show()
    