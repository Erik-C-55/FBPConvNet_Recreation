import os
import time
import torch
import random
import numpy as np
import torch.multiprocessing as mp

from sys import argv
from torchvision.utils import make_grid
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms import Compose, RandomHorizontalFlip, \
    RandomVerticalFlip

# Import from this repo
from lib.unet import Unet 
from lib.dataset import FBPDataset
# from lib.userInput import getUserOptions

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

def makeLoaders(options, gen):
    
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
    dataLoaders = makeLoaders(options, gen)
    
    if options.trainVal:
        tr_loader = dataLoaders[0]
        val_loader = dataLoaders[1]
    else:
        test_loader = dataLoaders[0]
           
    # Generate Model
    FBPConvNet = Unet(in_chans = 1, out_chans = 1)
    
    cur_time = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(time.time()))
    logdir = os.path.join('logs', cur_time)
    if not os.path.isdir(logdir):
        os.makedirs(logdir)
    
    # Generate Tensorboard SummaryWriter for tracking
    writer = SummaryWriter(log_dir=logdir)
    
    # If the user wants to add the model graph and sample images to Tensorboard
    if options.graph:
        if options.trainVal:
            low_fbp, full_fbp = iter(val_loader).next()
        else:
            low_fbp, full_fbp = iter(test_loader).next()
            
        writer.add_graph(FBPConvNet, low_fbp)
        writer.add_image('Low-View FBP', make_grid(low_fbp, pad_value=1.0,
                                                   normalize=False,
                                                   nrow=options.batch//2))
        writer.add_image("'Full-view' FBP", make_grid(full_fbp, pad_value=1.0,
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
    lr = 0.01 * float(options.batch)
    optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, FBPConvNet.parameters()),
                                lr=lr, momentum=0.99)
    
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, 0.977)
    
    valLoss = 1e5
    minValLoss = 1e5
    
    # # Train the model, if required
    # if options.trainVal:
    #     for epoch in range(1, options.max_epochs+1):
    #         train(FBPConvNet, tr_loader, epoch, loss, optimizer)
            
    #         if (epoch %2 ==0 or epoch == options.max_epochs):
    #             valLoss = float(validation(FBPConvNet, val_loader, epoch, loss))
                
    #             # Update the max tracker as needed
    #             if valLoss < minValLoss:
    #                 minValLoss = valLoss
                
    #                 # Don't worry about saving checkpoints until at
    #                 # least 1/5 through training, as it is unlikely 
    #                 # that the best model will be achieved before then
    #                 if (epoch > options.max_epochs//5):
    #                     checkpt_path = os.path.join(logdir, "epoch_" + str(epoch) + "_checkpoint.pth")
    #                     torch.save(FBPConvNet.state_dict(), checkpt_path)
                        
                        
    #         scheduler.step()
        
            
if __name__ == '__main__':
    
    # Handle torch backends for reproducibility
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    
    # Temp, for debug only
    from argparse import Namespace
    options = Namespace()
    options.n_ellipse = (5,14)
    options.workers = 4
    options.full_views = 1000
    options.low_views = 143
    options.n_samps = 500
    options.batch = 4
    options.trainVal = True
    options.graph = True
    options.max_epochs = 100

    # options = getUserOptions(argv)
    main(options)
    