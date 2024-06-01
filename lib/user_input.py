"""user_input.py

Contains an argparser to handle command-line arguments

"""

# Standard Imports
import os
import glob
import typing
import argparse

# Third-Party Imports
import numpy as np
import torch


def get_user_options(cl_args: typing.List[str]) -> argparse.Namespace:
    """Process command-line arguments using argparse.

    Parameters
    ----------
    cl_args : typing.List[str]
        Command-line arguments, obtained through sys.argv

    Returns
    -------
    args : argparse.Namespace
        Relevant options and arguments for execution
    """

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # Suppress filename & add required arguments
    parser.add_argument("filename", type=str, help=argparse.SUPPRESS)
    parser.add_argument('output_dir', type=str,
                        help='directory for holding dataset images')

    # Add optional arguments
    parser.add_argument('-b', '--batch', type=int, choices=(2 ** np.arange(6)),
                        help='batch size to use for model', default=8)
    parser.add_argument('-d', '--sched_decay', type=float, default=0.977,
                        help='learning rate decay factor')
    parser.add_argument('-e', '--max_epochs', type=int, default=100,
                        help='maximum number of training epochs')
    parser.add_argument('-g', '--graph_and_samples', action='store_true',
                        help='whether to add the model graph and sample ' +
                        'images to TensorBoard logs')
    parser.add_argument('-l', '--loss_function', type=str,
                        choices=['L1', 'MSE'], help='training loss function')
    parser.add_argument('-m', '--mode', type=str, default='train',
                        choices=['train', 'test', 'time_lapse'],
                        help='Mode in which to run the model. "train" trains' +
                        'a new model. "test" evaluates trained model ' +
                        'accuracy. "time_lapse" shows how reconstruction ' +
                        'on the test set improves as the model is trained')
    parser.add_argument('-n', '--n_ellipses', type=str, default='5-14',
                        choices=['5-14', '15-24', '25-34'],
                        help='upper and lower bounds on the number of ' +
                        'ellipses in each image')
    parser.add_argument('-p', '--pretrained_model', type=str, default=None,
                        help='filepath to the pretrained model to evaluate. ' +
                        'If not provided, will train a model from scratch.')
    parser.add_argument('-r', '--random_seed', type=int, default=0,
                        help='random seed to use for model training')
    parser.add_argument('-s', '--n_samples', type=int, choices=[500, 1000],
                        help='number of samples in the dataset')
    parser.add_argument('-t', '--tensorboard_logs', type=str, default='',
                        help='directory for tensorboard logs. If not ' +
                        'specified, will name automatically')
    parser.add_argument('-v', '--low_views', type=int, default=143,
                        choices=['50', '143'],
                        help='number of views for low-view reconstruction')

    args = parser.parse_args(cl_args)

    # Split number of ellipses
    temp = args.n_ellipses.split('-')
    args.n_ellipses = (int(temp[0]), int(temp(1)))

    # Define the loss function
    if args.loss_function == 'L1':
        args.loss_function = torch.nn.L1Loss()
    elif args.loss_function == 'MSE':
        args.loss_function = torch.nn.MSELoss()

    # Automatically generate log strings if nothing is passed
    if args.tensorboard_logs is None:
        if args.mode == 'train':
            args.tensorboard_logs = os.path.join(
                'logs', f'{args.n_ellipses[0]}_{args.low_views}_' +
                f'{args.n_samples}_{args.mode}')
        
        elif args.mode == 'test':
            tr_set = args.pretrained_model.split('_')
            ellipse_count = os.path.basename(tr_set[1])

            args.tensorboard_logs = os.path.join(
                'logs', f'trained_on_{ellipse_count}_{tr_set[2]}_' +
                f'{tr_set[3]}_test_on_{args.n_ellipses[0]}_{args.low_views}')
    
    # TODO: Explain this fancy searching in the help string
            # searchString = 'logs/Orig_Range/' + str(options.n_ellipse[0]) + '_' + \
    #    str(options.low_views) + '_' + str(options.n_samps) + '_train/*checkpoint.pth'
            
            # Perform testing for each time step along the way to show how training the
    # model improves reconstruction quality
    # for pretr_chkpt in glob(searchString):
    #     options.pretrained = pretr_chkpt

    #     main(options)

    # TODO: This is the string for testing
                #         # Take the last checkpoint, as it has lowest validation loss
                # cl_args.pretrained = glob(search_string)[-1]

    return args
