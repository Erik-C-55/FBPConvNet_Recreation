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
from lib.unet import UNet 
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

def train(model, tr_loader, criterion, optimizer, device):
    
    # Set the model to training mode
    model.train()
    
    total_loss = 0.0
    samplesSeen = 0.0
    
    for batch_idx, (low_fbp, full_fbp) in enumerate(tr_loader):
        
        # Zero the gradients
        optimizer.zero_grad()
        
        # Move inputs/labels to GPU
        low_fbp = low_fbp.to(device)
        full_fbp = full_fbp.to(device)
        
        # Reconstruct low-view images and compare to full-view images
        recon = model(low_fbp)
        loss = criterion(recon, full_fbp)
        
        # Backprop
        loss.backward()
               
        # Track running_loss
        # To handle the gradient clipping, I still want mean loss reduction.
        # For tracking purposes, since I have different batch sizes, I want
        # to sum them
        samplesSeen += len(recon)
        total_loss += (loss.item() * samplesSeen)
        
        # Clip the gradients, as suggested in the paper
        torch.nn.utils.clip_grad_value_(model.parameters(), clip_value=0.01)

        # Optimizer step
        optimizer.step()

        # Debug only
        if batch_idx ==0:
            print('total_loss: ' + str(total_loss))
    
    # Average total loss across samples
    total_loss = total_loss / samplesSeen
    
    print('Average Training Loss Per Sample: {:.4f}'.format(total_loss))
    
    return total_loss
        
def validation(model, val_loader, criterion, device):
    
    # Set model to eval mode
    model.eval()
    
    total_loss = 0.0
    samplesSeen = 0.0
    
    # Stop updating gradients for validation
    with torch.no_grad():
        
        for batch_idx, (low_fbp, full_fbp) in enumerate(val_loader):
            
            # Move inputs/labels to GPU
            low_fbp = low_fbp.to(device)
            full_fbp = full_fbp.to(device)
            
            # Reconstruct low-view images and compare to full-view images
            recon = model(low_fbp)
            loss = criterion(recon, full_fbp)
            
            # Track running_loss
            samplesSeen += len(recon)
            total_loss += (loss.item() * samplesSeen)
            
            # Debug only
            if batch_idx ==0:
                print('total_loss: ' + str(total_loss))
        
        # Average total loss across samples
        total_loss = total_loss/samplesSeen
        
        print('Average Validation Loss Per Sample: {:.4f}'.format(total_loss))
        
        return total_loss
    
def main(options, seed = 0):
    
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    
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
    FBPConvNet = UNet()
    
    cur_time = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(time.time()))
    logdir = os.path.join('logs', cur_time)
    if not os.path.isdir(logdir):
        os.makedirs(logdir)
    
    # Generate Tensorboard SummaryWriter for tracking
    writer = SummaryWriter(log_dir=logdir)
    
    # If the user wants to add the model graph and sample images to Tensorboard
    if options.graph:
        if options.trainVal:
            low_fbp, full_fbp = iter(tr_loader).next()
        else:
            low_fbp, full_fbp = iter(test_loader).next()
            
        writer.add_graph(FBPConvNet, low_fbp)
        writer.add_image('Low-View FBP', make_grid(low_fbp, pad_value=1.0,
                                                   normalize=False,
                                                   nrow=options.batch//2))
        writer.add_image("'Full-view' FBP", make_grid(full_fbp, pad_value=1.0,
                                                   normalize=False,
                                                   nrow=options.batch//2))
        
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
    
    valLoss = 0.5
    minValLoss = 0.5
    
    # Train the model, if required
    if options.trainVal:
        for epoch in range(1, options.max_epochs+1):
            print('---- Epoch ' + str(epoch) + ' ----')
            trLoss = float(train(FBPConvNet, tr_loader, loss, optimizer, device))
            
            # Check against validation images every 2nd epoch
            if (epoch %2 ==0 or epoch == options.max_epochs):
                valLoss = float(validation(FBPConvNet, val_loader, loss, device))
                
                # Update the max tracker as needed
                if valLoss < minValLoss:
                    minValLoss = valLoss
                
                    # Don't worry about saving checkpoints until at
                    # least 1/5 through training, as it is unlikely 
                    # that the best model will be achieved before then
                    if (epoch > options.max_epochs//5):
                        checkpt_path = os.path.join(logdir, "epoch_" + str(epoch) + "_checkpoint.pth")
                        torch.save(FBPConvNet.state_dict(), checkpt_path)
                        
            # Log info to SummaryWriter
            writer.add_scalars('Losses',{'Tr_Loss': trLoss, 'Val_Loss': valLoss},
                               epoch)
            scheduler.step()
    # TODO: Implement 'else' here
    
    hparam_dict = vars(options)
    # Convert n_ellipses back to a string
    hparam_dict['n_ellipse'] = str(hparam_dict['n_ellipse'])
    
    metric_dict = {'tr ' + options.loss + ' loss': trLoss,
                   'val ' + options.loss + ' loss': valLoss}
    
    writer.add_hparams(hparam_dict=hparam_dict, metric_dict=metric_dict)
    writer.close()
        
            
if __name__ == '__main__':
    
    # Handle torch backends for reproducibility
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    
    # Temp, for debug only
    from argparse import Namespace
    options = Namespace()
    
    # Fix constant options
    options.n_ellipse = (25,34)
    options.workers = 8
    options.full_views = 1000
    options.batch = 8
    options.trainVal = True
    options.max_epochs = 100
    options.loss = 'L1'
    
    # Iterate over other options being explored
    for lviews in [50]:
        for samps in [500]:
            
            options.low_views = lviews
            options.n_samps = samps
            
            # Only add graph if there are 500 samples
            if samps == 500:
                options.graph = True

    # options = getUserOptions(argv)
            main(options)
    
