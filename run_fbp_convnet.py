"""run_fbp_convnet.py

Runs the FBPConvNet model. For help, run `python run_fbp_convnet.py --help` in
the command-line"""

# Standard Imports
import os
import copy
import time
import random
import typing
import argparse
from sys import argv
from glob import glob

# Third-Party Imports
import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid
from torchvision.transforms import Compose, RandomHorizontalFlip, \
    RandomVerticalFlip

# Import from this repo
from lib.unet import UNet
from lib.dataset import FBPDataset, NUM_VIEWS_FULL
from lib.user_input import get_user_options

NUM_WORKERS = 8  # Number of dataloader workers


def seed_everything(seed: int) -> torch.Generator:
    """Sets the random seed for as many libraries as possible

    Parameters
    ----------
    seed : int
        The random seed to use
    """

    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    g = torch.Generator()
    g.manual_seed(seed)

    return g


def seed_worker(worker_id: int) -> None:
    """Random seeds for workers

    Parameters
    ----------
    worker_id : int
        id for this worker
    """
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def make_loaders(
        options: argparse.Namespace,
        gen: torch.Generator) -> typing.List[torch.utils.data.DataLoader]:
    """Makes dataloaders

    Parameters
    ----------
    options : argparse.Namespace
        Must have 'n_ellipse', 'low_views', 'n_samps', 'n_batch' elements
    gen : torch.Generator
        Generator for dataloaders

    Returns
    ----------
    [torch.utils.data.DataLoader]
        Either [training_dataloader, val_dataloader] or [test_dataloader] 
    """
    # Add data augmentation used by original authors
    tr_trnsfrm = Compose([RandomHorizontalFlip(p=0.5),
                          RandomVerticalFlip(p=0.5)])

    data_loaders = []

    if options.pretrained is None:
        # Get training dataset, dataloader
        tr_set = FBPDataset(n_ellipse=options.n_ellipse,
                            low_views=options.low_views, transform=tr_trnsfrm,
                            mode='train', n_samps=options.n_samps)

        tr_loader = torch.utils.data.DataLoader(tr_set,
                                                batch_size=options.batch,
                                                shuffle=True, pin_memory=True,
                                                num_workers=NUM_WORKERS,
                                                generator=gen,
                                                worker_init_fn=seed_worker)

        # Get validation dataset, dataloader
        val_set = FBPDataset(n_ellipse=options.n_ellipse,
                             low_views=options.low_views, transform=None,
                             mode='validation', n_samps=options.n_samps)

        val_loader = torch.utils.data.DataLoader(val_set,
                                                 batch_size=options.batch,
                                                 shuffle=False,
                                                 pin_memory=True,
                                                 num_workers=NUM_WORKERS)

        # Print dataset sizes for verification
        print('# of training images: ' + str(len(tr_set)))
        print('# of validation images: ' + str(len(val_set)))

        data_loaders = [tr_loader, val_loader]

    else:
        # Get testing dataset, dataloader
        test_set = FBPDataset(n_ellipse=options.n_ellipse,
                              low_views=options.low_views, transform=None,
                              mode='test', n_samps=options.n_samps)

        # For testing, load only one image at a time, so that nn.loss reduction
        # happens across all the pixels of an image, but not across multiple
        # batch images
        test_loader = torch.utils.data.DataLoader(test_set, batch_size=1,
                                                  shuffle=False,
                                                  pin_memory=True,
                                                  num_workers=NUM_WORKERS)

        # Print dataset size for verification
        print('# of testing images: ' + str(len(test_set)))

        data_loaders = [test_loader]

    return data_loaders


def train(model: torch.nn.Module, tr_loader: torch.utils.data.DataLoader,
          criterion: typing.Union[torch.nn.L1Loss, torch.nn.MSELoss],
          optimizer: torch.optim.Optimizer, device: torch.device,
          batch_size: int) -> float:

    # TODO: I don't think the return type is actually a float.  Check.
    """Trains the model

    Parameters
    ----------
    model : torch.nn.Module
        The model to be trained
    tr_loader : torch.utils.data.DataLoader
        DataLoader for training data
    criterion : torch.nn.L1Loss or torch.nn.MSELoss
        Training loss function
    optimizer : torch.optim.Optimizer
        The optimizer to use
    device : torch.device
        The GPU or CPU device
    batch_size : int
        The training batch size

    Returns
    ----------
    loss : float
        Average loss across all batches
    """

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

    print(f'Average Training Loss Per Sample: {total_loss:.6f}')

    return total_loss


def validation(model: torch.nn.Module, val_loader: torch.data.utils.DataLoader,
               criterion: typing.Union[torch.nn.L1Loss, torch.nn.MSELoss],
               device: torch.device, batch_size: int) -> float:
    # TODO: I don't think the return type is actually a float.  Check.
    """Trains the model

    Parameters
    ----------
    model : torch.nn.Module
        The model to be trained
    val_loader : torch.utils.data.DataLoader
        DataLoader for validation data
    criterion : torch.nn.L1Loss or torch.nn.MSELoss
        Validation loss function
    optimizer : torch.optim.Optimizer
        The optimizer to use
    device : torch.device
        The GPU or CPU device
    batch_size : int
        The training batch size

    Returns
    ----------
    loss : float
        Average loss across all batches
    """

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

        print(f'Average Validation Loss Per Sample: {total_loss:.6f}')

        return total_loss


def test(model: torch.nn.Module, test_loader: torch.utils.data.DataLoader,
         device: torch.device):
    # TODO: I don't think the return type is actually a float.  Check.
    """Tests the model

    Parameters
    ----------
    model : torch.nn.Module
        The model to be trained
    test_loader : torch.utils.data.DataLoader
        DataLoader for test data
    device : torch.device
        The GPU or CPU device

    Returns
    ----------
    loss : float
        Average loss across all batches
    """
    # Set model to eval mode
    model.eval()

    # Initialize performance tensors
    mse_loss = torch.nn.MSELoss()
    fbp_mse = []
    fbp_psnr = []
    unet_mse = []
    unet_psnr = []

    # Save a sample return tensor
    recon_samp = torch.zeros((512, 512))

    # Don't update gradients in test
    with torch.no_grad():

        for batch_idx, (low_fbp, full_fbp) in enumerate(test_loader):

            # Move inputs/ground truth to GPU
            low_fbp = low_fbp.to(device)
            full_fbp = full_fbp.to(device)

            # Reconstruct low-view images
            recon = model(low_fbp)

            # Add the FBP mse and U-Net mse to their respective tensor lists
            fbp_mse.append(mse_loss(low_fbp, full_fbp).view(1))
            unet_mse.append(mse_loss(recon, full_fbp).view(1))

            # Calculate the squared max of the ground truth
            truth_max_sq = torch.square(torch.max(full_fbp))

            # Calculate PSNR = 10log10((max(truth)^2)/MSE)
            fbp_psnr.append(10*torch.log10(
                torch.divide(truth_max_sq, fbp_mse[batch_idx])).view(1))
            unet_psnr.append(10*torch.log10(
                torch.divide(truth_max_sq, unet_mse[batch_idx])).view(1))

            # Save the first reconstructed image as an example
            if batch_idx == 0:
                recon_samp = recon

        # Combine tensor lists using cat
        fbp_mse = torch.cat(fbp_mse, dim=0)
        fbp_psnr = torch.cat(fbp_psnr, dim=0)
        unet_mse = torch.cat(unet_mse, dim=0)
        unet_psnr = torch.cat(unet_psnr, dim=0)

        print(f'Average MSE per FBP sample: {torch.mean(fbp_mse).item()}')
        print(f'Average PSNR per FBP sample: {torch.mean(fbp_psnr).item()}')
        print(f'Average MSE per UNet sample: {torch.mean(unet_mse).item()}')
        print(f'Average PSNR per UNet sample: {torch.mean(unet_psnr).item()}')

        return fbp_mse, fbp_psnr, unet_mse, unet_psnr, recon_samp


def main(args):

    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    # Seed things for reproducibility
    gen = seed_everything(args.seed)

    # Instantiate dataloaders, depending on testing or training setting
    data_loaders = make_loaders(args, gen)

    if args.pretrained is None:
        tr_loader = data_loaders[0]
        val_loader = data_loaders[1]
    else:
        test_loader = data_loaders[0]

    # Generate Model
    fbp_convnet = UNet()

    # Generate the appropriate log directory
    if args.mode == 'train':
        logdir = os.path.join(
            'logs',
            f'{args.n_ellipse[0]}_{args.low_views}_{args.n_samps}_{args.mode}')

    elif args.mode == 'test':
        # Extract the info on the training data from the checkpoint path
        train_set = args.pretrained.split('_')
        ellipse_count = train_set[1].split(os.path.sep)[-1]

        dir_string = f'trained_on_{ellipse_count}_{train_set[2]}_' + \
            f'{train_set[3]}_test_on_{args.n_ellipse[0]}_{args.low_views}'

        logdir = os.path.join('logs', dir_string)

    elif args.mode == 'time_lapse':
        # Extract the trainin epoch number from the checkpoint path
        tr_epoch = args.pretrained.split('_')[-2]
        logdir = os.path.join('logs', str(args.n_ellipse[0]) + '_' +
                              str(args.low_views) + '_' + str(args.n_samps) +
                              '_' + args.mode + '_epoch_' + tr_epoch)

    if not os.path.isdir(logdir):
        os.makedirs(logdir)
    else:
        print('A log directory for this setting exists. ' +
              'Using timestamp as directory name.')
        cur_time = time.strftime(
            '%Y-%m-%d-%H-%M-%S', time.localtime(time.time()))
        logdir = os.path.join('logs', cur_time)
        os.makedirs(logdir)

    # Generate Tensorboard SummaryWriter for tracking
    writer = SummaryWriter(log_dir=logdir)

    # If the user wants to add the model graph and sample images to Tensorboard
    if args.graph:
        if args.pretrained is None:
            low_fbp, full_fbp = iter(tr_loader).next()
        else:
            low_fbp, full_fbp = iter(test_loader).next()

        # Turned off after the first time, now that I have a model graph
        # writer.add_graph(FBPConvNet, low_fbp)

        # Add images to tensorboard. Value_range clamps and normalizes
        writer.add_image('Low-View FBP (Model Input)',
                         make_grid(low_fbp, pad_value=0.0, normalize=True,
                                   value_range=(-500.0, 500.0),
                                   nrow=args.batch//2))
        writer.add_image("'Full-view' FBP (Ground Truth)",
                         make_grid(full_fbp, pad_value=0.0, normalize=True,
                                   value_range=(-500.0, 500.0),
                                   nrow=args.batch//2))

    # Train the model, if required
    if args.pretrained is None:

        # Move the model to the GPU
        fbp_convnet = fbp_convnet.to(device)

        # Define the loss, optimizer, and scheduler
        if args.loss == 'L1':
            loss = torch.nn.L1Loss()
        elif args.loss == 'MSE':
            loss = torch.nn.MSELoss()

        # Original authors used batch=1. If batchsize is greater, scale lr
        lr = 0.01 * float(args.batch)
        optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad,
                                           fbp_convnet.parameters()), lr=lr,
                                    momentum=0.99)

        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer,
                                                           args.sched_decay)

        val_loss = 15
        min_val_loss = 15

        for epoch in range(1, args.max_epochs+1):
            print('---- Epoch ' + str(epoch) + ' ----')
            tr_loss = float(train(fbp_convnet, tr_loader, loss, optimizer,
                                  device, args.batch))

            # Check against validation images every 2nd epoch
            if (epoch % 2 == 0 or epoch == args.max_epochs):
                val_loss = float(validation(fbp_convnet, val_loader, loss,
                                            device, args.batch))

                # Update the max tracker as needed
                if val_loss < min_val_loss:
                    min_val_loss = val_loss

                    # Don't worry about saving checkpoints until at
                    # least 1/5 through training, as it is unlikely
                    # that the best model will be achieved before then
                    if epoch > args.max_epochs//5:
                        checkpt_path = os.path.join(
                            logdir, f'epoch_{epoch}_checkpoint.pth')
                        torch.save(fbp_convnet.state_dict(), checkpt_path)

            # Log info to SummaryWriter
            writer.add_scalars('Losses', {'Tr_Loss': tr_loss,
                                          'Val_Loss': val_loss}, epoch)
            scheduler.step()

        # Generate train-appropriate metric dictionary
        metric_dict = {'tr ' + args.loss + ' loss': tr_loss,
                       'val ' + args.loss + ' loss': val_loss}

    # Test the model if pretrained weights are passed
    else:

        # Load pretrained weights
        pretrain_dict = torch.load(args.pretrained, map_location='cpu')
        try:
            model_dict = fbp_convnet.state_dict()
        except AttributeError:
            model_dict = fbp_convnet.state_dict()
        pretrain_dict = {k: v for k,
                         v in pretrain_dict.items() if k in model_dict.keys()}

        print("Using file " + args.pretrained + ' to load ' +
              str(len(pretrain_dict)) + ' of the ' + str(len(model_dict)) +
              ' tensors of parameters in the model')

        model_dict.update(pretrain_dict)
        fbp_convnet.load_state_dict(model_dict)

        # Move the model to the GPU
        fbp_convnet = fbp_convnet.to(device)

        fbp_mse, fbp_psnr, unet_mse, unet_psnr, recon_samp = test(fbp_convnet,
                                                                  test_loader,
                                                                  device)

        # Add the reconstructed image to tensorboard files
        writer.add_image(f"UNet Output. MSE: {unet_mse[0]:.4f}. " +
                         "PSNR: {unet_psnr[0]:.4f}",
                         make_grid(recon_samp, pad_value=0.0, normalize=True,
                                   value_range=(-500.0, 500.0),
                                   nrow=args.batch//2))

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

    hparam_dict = vars(copy.deepcopy(args))
    # Convert n_ellipses back to a string
    hparam_dict['n_ellipse'] = str(hparam_dict['n_ellipse'])
    hparam_dict['full_views'] = NUM_VIEWS_FULL

    writer.add_hparams(hparam_dict=hparam_dict, metric_dict=metric_dict)
    writer.close()


if __name__ == '__main__':

    # Handle torch backends for reproducibility
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    # Temp, for debug only
    from argparse import Namespace
    cl_args = Namespace()

    # Fix constant options
    cl_args.batch = 8
    cl_args.pretrained = None
    cl_args.max_epochs = 100
    cl_args.loss = 'L1'
    cl_args.sched_decay = 0.977
    cl_args.seed = 0
    cl_args.graph = False

    # Training setup ---------------------------------------------------------
    # options.mode = 'train'
    # #  Iterate over other options being explored
    # for n_ellipse in [(25,34)]: # [(5,14),(15,24),(25,34)]:
    #     for lviews in [143]: #[50,143]:
    #         for samps in [1000]: #[500,1000]:

    #            options.n_ellipse = n_ellipse
    #            options.low_views = lviews
    #            options.n_samps = samps

    #            # Only add graph if there are 500 samples
    #            if samps == 500 and options.pretrained is None:
    #                options.graph = True
    #            else:
    #                options.graph = False

    #            # options = getUserOptions(argv)
    #            main(options)

    # Time-Lapse Setup --------------------------------------------------------
    # options.graph = True
    # options.mode = 'time_lapse'
    # options.batch = 4
    # options.n_ellipse = (15,24)
    # options.low_views = 50
    # options.n_samps = 1000

    # Generate the appropriate weights file automatically
    # searchString = 'logs/Orig_Range/' + str(options.n_ellipse[0]) + '_' + \
    #    str(options.low_views) + '_' + str(options.n_samps) + '_train/*checkpoint.pth'

    # Perform testing for each time step along the way to show how training the
    # model improves reconstruction quality
    # for pretr_chkpt in glob(searchString):
    #     options.pretrained = pretr_chkpt

    #     main(options)

    # Cross-Testing Setup -----------------------------------------------------
    cl_args.graph = True
    cl_args.mode = 'test'

    #  Iterate over other options being explored
    for n_ellipse in [(5, 14), (15, 24), (25, 34)]:
        for lviews in [50, 143]:
            for samps in [500, 1000]:

                # Generate the appropriate weights file automatically
                search_string = f'logs/Orig_Range/{n_ellipse[0]}_' + \
                    str(lviews) + '_' + str(samps) + '_train/*checkpoint.pth'

                # Take the last checkpoint, as it has lowest validation loss
                cl_args.pretrained = glob(search_string)[-1]
                cl_args.n_samps = samps

                # Using the weights file, iterate over all 6 test combinations
                # for this number of samples and these weights
                for ellipses in [(5, 14), (15, 24), (25, 34)]:
                    for low_views in [50, 143]:

                        print(f'Testing on {ellipses} ellipses with ' +
                              f'{low_views} views.')

                        cl_args.n_ellipse = ellipses
                        cl_args.low_views = low_views

                        main(cl_args)
