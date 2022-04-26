import os
import torch
import random
import numpy as np

from sys import argv
from torchvision.transforms import Compose, RandomHorizontalFlip, \
    RandomVerticalFlip

# Import from this repo
from lib.unet import Unet 
from lib.dataset import FBPDataset
from lib.userInput import getUserOptions

def seedEverything(seed):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    g = torch.Generator()
    g.manual_seed(seed)
    
    return g

def loadData(options, gen):
    
    # Add data augmentation used by original authors
    tr_transform = Compose([RandomHorizontalFlip(p=0.5),
                            RandomVerticalFlip(p=0.5)])
    
    tr_set = FBPDataset(n_ellipse=options.n_ellipse, im_size=options.im_size,
                        full_views=options.full_views,
                        low_views=options.low_views, transform=tr_transform,
                        mode='train', n_samps=options.n_samps)
    
def main(options, seed = 0):
    
    # Seed things for reproducibility
    gen = seedEverything(seed)
    
    # Generate Model
    model = Unet(in_chans = 1, out_chans = 1)

if __name__ == '__main__':
    options = getUserOptions(argv)
    main(options)
    