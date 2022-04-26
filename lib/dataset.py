import os
import torch
import torchvision
import numpy as np
import matplotlib.pyplot as plt

# Import Parallel Beam Radon transform function
from torch_radon import Radon
from torch.utils.data import DataLoader, Dataset

class FBPDataset(Dataset):
    
    def __init__(self, directory, im_size=512, full_views=1000, low_views=143,
                 transform=None, mode=train):
        self.directory = directory
        self.im_size = im_size
        self.full_views = full_views
        self.low_views = low_views
        self.transform = transform
        self.mode = mode
        self.fnames = []
        
    def obtainImList(self):
        # Open the appropriate .txt key file and read all the lines
        keyFile = open(os.path.abspath(os.path.join(os.path.dirname(__file__),
                                                    os.pardir,'imageData',
                                                # Handle file name strings    'Val_'+str(valPerc)+'_Seed_0',self.mode+'.txt')),'r')
        keyList = keyFile.readlines()
        keyFile.close()
        
        # Iterate over list, strip newline character,and add it to
        # list of files/labels in this dataset
        for im in keyList:
            
            self.fnames.append(os.path.join(self.directory,im.rstrip('\n')))

        
    def __getitem__(self, index):
        origIm = np.load(self.fnames[index])
        
        if self.transform is not None:
            origIm = self.transform(origIm)
        
        # Calculate all view angles
        angles = np.linspace(0, np.pi, self.full_views, endpoint=False)

        # Calculate detector count (default from torch_radon example)
        det_count = int(np.sqrt(2)*self.im_size + 0.5)
        
        # Generate Radon transform for full-view parallel-beam CT
        radon = Radon(self.im_size, angles, clip_to_circle=False,
                      det_count=det_count)
        
        # Calculate full-view sinogram and filtered backprojection
        full_sgram = radon.forward(origIm)
        full_fbp = radon.backprojection(radon.filter_sinogram(full_sgram,
                                                             filter_name='ramp'))
        
        # Calculate all view angles
        angles = np.linspace(0, np.pi, self.low_views, endpoint=False)
        
        # Generate Radon transform for low-view parallel-beam CT
        radon = Radon(self.im_size, angles, clip_to_circle=False,
                      det_count=det_count)
        
        # Calculate low-view sinogram and filtered backprojection
        low_sgram = radon.forward(origIm)
        low_fbp = radon.backprojection(radon.filter_sinogram(low_sgram,
                                                             filter_name='ramp'))
        
        return low_fbp, full_fbp, low_sgram, full_sgram
       
if __name__ == 'main':
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
    