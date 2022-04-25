import os
import torch
import torchvision
import numpy as np

# Import Parallel Beam Radon transform function
from torch_radon import Radon
from torch.utils.data import DataLoader, Dataset

class FBPDataset(Dataset):
    
    def __init__(self, directory, im_size=512, views=1000, transform=None):
        self.directory = directory
        self.im_size = im_size
        self.views = views
        self.transform = transform
        self.fnames = []
        
    def obtainImList(self):
        # Open the appropriate .txt key file and read all the lines
        keyFile = open(os.path.abspath(os.path.join(os.path.dirname(__file__),
                                                    os.pardir,'dataPartitions',
                                                    self.datasetName,
                                                    'Val_'+str(valPerc)+'_Seed_0',self.mode+'.txt')),'r')
        keyList = keyFile.readlines()
        keyFile.close()
        
        # Iterate over list, strip newline character,and add it to
        # list of files/labels in this dataset
        for im in keyList:
            
            self.fnames.append(os.path.join(self.directory,im.rstrip('\n')))

        
    def __getitem__(self, index):
        fullViewIm = np.load(fnames[index])
        lowViewIm = np.load(fnames[index])
        
        if self.transform is not None:
            