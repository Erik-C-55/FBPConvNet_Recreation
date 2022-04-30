import os
import time
import torch
import random
import numpy as np

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
    
    if options.pretrained is None:
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
                            low_views=options.low_views, transform=None,
                            mode='test', n_samps=options.n_samps)
        
        # For testing, load only one image at a time, so that nn.loss reduction
        # happens across all the pixels of an image, but not across multiple
        # batch images
        test_loader=torch.utils.data.DataLoader(test_set, batch_size=1,
                                              shuffle=False, pin_memory=True,
                                              num_workers=options.workers)
        
        # Print dataset size for verification
        print('# of testing images: ' + str(len(test_set)))
        
        dataLoaders = [test_loader]
        
    return dataLoaders

def train(model, tr_loader, criterion, optimizer, device, batch_size):
    
    # Set the model to training mode
    model.train()
    
    total_loss = 0.0
    
    for batch_idx, (low_fbp, full_fbp) in enumerate(tr_loader):
        
        # Zero the gradients
        optimizer.zero_grad()
        
        # Move inputs/ground truth to GPU
        low_fbp = low_fbp.to(device)
        full_fbp = full_fbp.to(device)
        
        # Reconstruct low-view images and compare to full-view images
        recon = model(low_fbp)
        loss = criterion(recon, full_fbp)
        
        # Backprop
        loss.backward()
        
        # Loss is averaged across batch. Adjust scaling for small batches
        prop_full_batch = float(len(recon))/float(batch_size)
        total_loss += (loss.item() * prop_full_batch)
        
        # Clip the gradients, as suggested in the paper
        torch.nn.utils.clip_grad_value_(model.parameters(), clip_value=0.01)

        # Optimizer step
        optimizer.step()
    
    # Average total loss across batches, including the small last batch
    total_loss = total_loss / (batch_idx + prop_full_batch)
    
    print('Average Training Loss Per Sample: {:.6f}'.format(total_loss))
    
    return total_loss
        
def validation(model, val_loader, criterion, device, batch_size):
    
    # Set model to eval mode
    model.eval()
    
    total_loss = 0.0
    
    # Stop updating gradients for validation
    with torch.no_grad():
        
        for batch_idx, (low_fbp, full_fbp) in enumerate(val_loader):
            
            # Move inputs/ground truth to GPU
            low_fbp = low_fbp.to(device)
            full_fbp = full_fbp.to(device)
            
            # Reconstruct low-view images and compare to full-view images
            recon = model(low_fbp)
            loss = criterion(recon, full_fbp)
            
            # Loss is averaged across batch. Adjust scaling for small batches
            prop_full_batch = float(len(recon))/float(batch_size)
            total_loss += (loss.item() * prop_full_batch)
        
        # Average total loss across batches, including the small last batch
        total_loss = total_loss / (batch_idx + prop_full_batch)
        
        print('Average Validation Loss Per Sample: {:.6f}'.format(total_loss))
        
        return total_loss
    
def test(model, test_loader, device):
    
    # Set model to eval mode
    model.eval()
    
    # Initialize performance tensors
    mseLoss = torch.nn.MSELoss()
    fbp_mse = []
    fbp_psnr = []
    unet_mse = []
    unet_psnr = []
    
    # Save a sample return tensor
    sampleRecon = torch.zeros((512,512))
    
    # Don't update gradients in test
    with torch.no_grad():
        
        for batch_idx, (low_fbp, full_fbp) in enumerate(test_loader):
            
            # Move inputs/ground truth to GPU
            low_fbp = low_fbp.to(device)
            full_fbp = full_fbp.to(device)
            
            # Reconstruct low-view images
            recon = model(low_fbp)
            
            # Add the FBP mse and U-Net mse to their respective tensor lists
            fbp_mse.append(mseLoss(low_fbp, full_fbp).view(1))
            unet_mse.append(mseLoss(recon, full_fbp).view(1))
            
            # Calculate the squared max of the ground truth
            truth_max_sq = torch.square(torch.max(full_fbp))
            
            # Calculate PSNR = 10log10((max(truth)^2)/MSE)
            fbp_psnr.append(10*torch.log10(torch.divide(truth_max_sq,
                                                             fbp_mse[batch_idx])).view(1))            
            unet_psnr.append(10*torch.log10(torch.divide(truth_max_sq,
                                                             unet_mse[batch_idx])).view(1))
            
            # Save the first reconstructed image as an example
            if batch_idx == 0:
                sampleRecon = recon
        
        # Combine tensor lists using cat
        fbp_mse = torch.cat(fbp_mse, dim=0)
        fbp_psnr = torch.cat(fbp_psnr, dim=0)
        unet_mse = torch.cat(unet_mse, dim=0)
        unet_psnr = torch.cat(unet_psnr, dim=0)

        print('Average MSE per FBP sample: ' + str(torch.mean(fbp_mse)))
        print('Average PSNR per FBP sample: ' + str(torch.mean(fbp_psnr)))
        print('Average MSE per UNet sample: ' + str(torch.mean(unet_mse)))
        print('Average PSNR per UNet sample: ' + str(torch.mean(unet_psnr)))
    
        return fbp_mse, fbp_psnr, unet_mse, unet_psnr, sampleRecon
    
def main(options):
    
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    
    # Seed things for reproducibility
    gen = seedEverything(options.seed)
    
    # Instantiate dataloaders, depending on testing or training setting
    dataLoaders = makeLoaders(options, gen)
    
    if options.pretrained is None:
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
        if options.pretrained is None:
            low_fbp, full_fbp = iter(tr_loader).next()
        else:
            low_fbp, full_fbp = iter(test_loader).next()
            
        # Turned off after the first time, now that I have a model graph
        # writer.add_graph(FBPConvNet, low_fbp)
        
        writer.add_image('Low-View FBP (Model Input)',
                         make_grid(low_fbp, pad_value=0.0, normalize=False,
                                   nrow=options.batch//2))
        writer.add_image("'Full-view' FBP (Ground Truth)",
                         make_grid(full_fbp, pad_value=0.0, normalize=False,
                                   nrow=options.batch//2))
    
    # Train the model, if required
    if options.pretrained is None:
        
        # Move the model to the GPU
        FBPConvNet = FBPConvNet.to(device)
        
        # Define the loss, optimizer, and scheduler
        if options.loss == 'L1':
            loss = torch.nn.L1Loss()
        elif options.loss == 'MSE':
            loss = torch.nn.MSELoss()
            
        # Original authors used batchsize of 1.  If batchsize is greater, scale lr
        lr = 0.01 * float(options.batch)
        optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad,
                                           FBPConvNet.parameters()), lr=lr,
                                    momentum=0.99)
        
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer,
                                                           options.sched_decay)
        
        valLoss = 0.09
        minValLoss = 0.09
        
        for epoch in range(1, options.max_epochs+1):
            print('---- Epoch ' + str(epoch) + ' ----')
            trLoss = float(train(FBPConvNet, tr_loader, loss, optimizer, device,
                                 options.batch))
            
            # Check against validation images every 2nd epoch
            if (epoch %2 ==0 or epoch == options.max_epochs):
                valLoss = float(validation(FBPConvNet, val_loader, loss, device,
                                           options.batch))
                
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
        
        # Generate train-appropriate metric dictionary
        metric_dict = {'tr ' + options.loss + ' loss': trLoss,
                       'val ' + options.loss + ' loss': valLoss}
            
    # Test the model if pretrained weights are passed
    else:
        
        # Load pretrained weights
        pretrained_dict = torch.load(options.pretrained, map_location='cpu')
        try:
            model_dict = FBPConvNet.state_dict()
        except AttributeError:
            model_dict = FBPConvNet.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}

        print("Using file " + options.pretrained + ' to load ' + \
              str(len(pretrained_dict)) + ' of the ' + str(len(model_dict)) + \
                  ' tensors of parameters in the model')
 
        model_dict.update(pretrained_dict)
        FBPConvNet.load_state_dict(model_dict)
        
        # Move the model to the GPU
        FBPConvNet = FBPConvNet.to(device)
        
        fbp_mse, fbp_psnr, unet_mse, unet_psnr, sampleRecon = test(FBPConvNet,
                                                                   test_loader,
                                                                   device)
        
        # Add the reconstructed image to tensorboard files
        writer.add_image("UNet Output Image. MSE: {:.4f}. PSNR: {:.4f}".format(unet_mse[0], unet_psnr),
                         make_grid(sampleRecon, pad_value=0.0, normalize=False,
                                   nrow=options.batch//2))
        
        # Save tensors describing metrics
        torch.save(fbp_mse, os.path.join(logdir, 'fbp_mse.pth'))
        torch.save(fbp_psnr, os.path.join(logdir, 'fbp_psnr.pth'))
        torch.save(unet_mse, os.path.join(logdir, 'unet_mse.pth'))
        torch.save(unet_psnr, os.path.join(logdir, 'unet_psnr.pth'))
        
        # Generate test-appropriate metric dictionary
        metric_dict = {'Mean FBP MSE': torch.mean(fbp_mse),
                       'Mean FBP PSNR': torch.mean(fbp_psnr),
                       'Mean UNet MSE': torch.mean(unet_mse),
                       'Mean UNet PSNR': torch.mean(unet_psnr)}
                       
        
    hparam_dict = vars(options)
    # Convert n_ellipses back to a string
    hparam_dict['n_ellipse'] = str(hparam_dict['n_ellipse'])
    
    writer.add_hparams(hparam_dict=hparam_dict, metric_dict=metric_dict)
    writer.close()
        
            
if __name__ == '__main__':
    
    # Handle torch backends for reproducibility
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    
    # Temp, for debug only
    from argparse import Namespace
    options = Namespace()
    
    # Fix constant options
    options.workers = 8
    options.full_views = 1000
    options.batch = 8
    options.pretrained = 'logs/Clipping/2022-04-30-10-35-08/epoch_96_checkpoint.pth'
    options.max_epochs = 100
    options.loss = 'L1'
    options.sched_decay = 0.977
    options.seed = 0

    # Iterate over other options being explored
    # for n_ellipse in [(5,14),(15,24),(25,34)]:
    #    for lviews in [50,143]:
    #        for samps in [500,1000]:
                
    options.n_ellipse = (25,34)
    options.low_views = 143
    options.n_samps = 1000
                
    # Only add graph if there are 500 samples
    # if samps == 500 or pretrained is not None:
    options.graph = True
    
    # options = getUserOptions(argv)
    main(options)
    
