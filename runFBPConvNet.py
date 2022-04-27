import os
import torch
import random
import numpy as np

from sys import argv
from torchvision.utils import make_grid
from torch.utils.tensorboard import SummaryWriter
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

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def loadData(options, gen):
    
    # Add data augmentation used by original authors
    tr_trnsfrm = Compose([RandomHorizontalFlip(p=0.5),
                            RandomVerticalFlip(p=0.5)])
    dataLoaders = []
    
    if options.trainVal:
        # Get training dataset, dataloader
        tr_set = FBPDataset(n_ellipse=options.n_ellipse,
                            full_views=options.full_views,
                            low_views=options.low_views, transform=tr_trnsfrm,
                            mode='train', n_samps=options.n_samps)
        
        tr_loader=torch.utils.data.DataLoader(tr_set, batch_size=options.batch,
                                              shuffle=True, pin_memory=True,
                                              num_workers=options.workers,  
                                              generator=gen,
                                              worker_init_fn=seed_worker)
        
        # Get validation dataset, dataloader
        val_set = FBPDataset(n_ellipse=options.n_ellipse,
                            full_views=options.full_views,
                            low_views=options.low_views, transform=None,
                            mode='validation', n_samps=options.n_samps)
        
        val_loader=torch.utils.data.DataLoader(val_set, batch_size=options.batch,
                                              shuffle=False, pin_memory=True,
                                              num_workers=options.workers)
    
        # Print dataset sizes for verification
        print('# of training images: ' + str(len(tr_set)))
        print('# of validation images: ' + str(len(val_set)))
        
        dataLoaders = [tr_loader, val_loader]
        
    else:
        # Get testing dataset, dataloader
        test_set = FBPDataset(n_ellipse=options.n_ellipse,
                            full_views=options.full_views,
                            low_views=options.low_views, transform=tr_trnsfrm,
                            mode='train', n_samps=options.n_samps)
        
        test_loader=torch.utils.data.DataLoader(test_set, batch_size=options.batch,
                                              shuffle=False, pin_memory=True,
                                              num_workers=options.workers)
        
        # Print dataset size for verification
        print('# of testing images: ' + str(len(test_set)))
        
        dataLoaders = [test_loader]
        
    return dataLoaders

    
def main(options, seed = 0):
    
    if torch.cuda.is_available():
        device = torch.device('cuda')
    
    # Seed things for reproducibility
    gen = seedEverything(seed)
    
    # Instantiate dataloaders, depending on testing or training setting
    dataLoaders = loadData(options, gen)
    
    if options.trainVal:
        tr_loader = dataLoaders[0]
        val_loader = dataLoaders[1]
    else:
        test_loader = dataLoaders[0]
           
    # Generate Model
    FBPConvNet = Unet(in_chans = 1, out_chans = 1)
    
    # Generate Tensorboard SummaryWriter for tracking
    writer = SummaryWriter(log_dir='logs')
    
    # If the user wants to add the model graph and sample images to Tensorboard
    if options.graph:
        if options.trainVal:
            low_fbp, full_fbp, low_sgram, full_sgram = iter(val_loader).next()
        else:
            low_fbp, full_fbp, low_sgram, full_sgram = iter(test_loader).next()
            
        writer.add_graph(FBPConvNet, low_fbp)
        writer.add_image('Low-View FBP', make_grid(low_fbp, pad_value=1.0,
                                                   normalize=False,
                                                   nrow=options.batch//2))
        writer.add_image("'Full-view' FBP", make_grid(full_fbp, pad_value=1.0,
                                                   normalize=False,
                                                   nrow=options.batch//2))
        writer.add_image('Low-View Sinogram', make_grid(low_sgram, pad_value=1.0,
                                                   normalize=False,
                                                   nrow=options.batch//2))
        writer.add_image("'Full-View' Sinogram", make_grid(full_sgram, pad_value=1.0,
                                                   normalize=False,
                                                   nrow=options.batch//2))
        # TIP: Debugging only
        writer.close()
        
    # Move the model to the GPU
    FBPConvNet = FBPConvNet.to(device)
    
    # Define the loss, optimizer, and scheduler
    if options.loss == 'L1':
        loss = torch.nn.L1Loss()
    elif options.loss == 'MSE':
        loss = torch.nn.MSELoss()
        
    # Original authors used batchsize of 1.  If batchsize is greater, scale lr
    lr = 
    optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, FBPConvNet.parameters()),
                                lr=0.01, momentum)
        
    
    

if __name__ == '__main__':
    options = getUserOptions(argv)
    main(options)
    