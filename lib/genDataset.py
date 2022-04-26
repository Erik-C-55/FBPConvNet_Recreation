import os
import torch
import random
import argparse
import numpy as np
import matplotlib.pyplot as plt

from sys import argv
from scipy.stats import mode

# Import functions from this repo
from ellipseGenerator import genEllipse

# Import Parallel Beam Radon transform function
from torch_radon import Radon

# TODO: Just make the dataset very rigid, so they always know there will be 1500
# of each type of image and that they will follow my folder names.  It makes 
# things more accesible.

# This function parses command-line arguments to allow flexible execution.  For
# an explanation 
def getArgs(CLArgs):
    """This function uses a Python argument parser to process command-line
    arguments.  For an explanation of each command-line option, see the 
    associated help string in the parser.add_argument() call.
    
    Parameters:
        CLArgs: A list of command-line arguments, obtained through sys.argv
        
    Returns:
        args: A Namespace object containing relevant options and arguments for
            execution.  For a list of available options and arguments, see the
            parser.add_argument() calls below
    """
    parser = argparse.ArgumentParser()
    
    # Suppress filename & add required arguments
    parser.add_argument("filename",type=str,help=argparse.SUPPRESS)
    parser.add_argument('output_dir', type=str,
                        help='directory for holding dataset images')
    
    # Add optional arguments
    parser.add_argument('-d','--display', action='store_true',
                        help='display samples of the first image generated. If unspecified, defaults to False.')
    parser.add_argument('-l','--lower_bound', type=int, default=5,
                        help='a random number (integer) of ellipses will be generated.  This is the lower bound for that random number. Default is 5.')
    parser.add_argument('-r','--res', type=int, default=512,
                        help='image resolution in pixels (integer).  Image will be square (res x res).  Resolution should be a multiple of 16. Default is 512.')
    parser.add_argument('-s','--samples', type=int, default=10,
                        help='integer number of sample images to generate. Default is 10.')
    parser.add_argument('-u','--upper_bound', type=int, default=15,
                        help='a random number (integer) of ellipses will be generated.  This is the upper bound for that random number. Default is 15')
    
    args = parser.parse_args(CLArgs)
    
    # Ensure that the upper bound is greater than the lower bound
    if args.lower_bound > args.upper_bound:
        args.upper_bound = args.lower_bound + 10
    
    return args

def applyRadon(im, idx, directory, n_views=[1000,143,50], display=False):
    
    # Move images to GPU for speed
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print('CUDA is available!')
        
    im = torch.FloatTensor(im).to(device)
        
    for views in n_views:
        # Generate name for directory.  If it doesn't exist, create it
        destDir = os.path.join(directory, str(views) + '_views')
        
        if not os.path.isdir(destDir):
            os.mkdir(destDir)
      
        im_size = im.size()[0]
        
        # Calculate all view angles for 1000-view projection
        angles = np.linspace(0, np.pi, 1000, endpoint=False)
    
        # Calculate detector count (default from torch_radon example)
        det_count = int(np.sqrt(2)*im_size + 0.5)
        

        # Generate Radon transform for full-view parallel-beam CT
        radon = Radon(im_size, angles, clip_to_circle=False,
                      det_count=det_count)
        
        # Calculate full-view sinogram and filtered backprojection
        sgram = radon.forward(im)
        fbp = radon.backprojection(radon.filter_sinogram(sgram,
                                                         filter_name='ramp'))
        
        # If displaying, display for index 1
        if idx == 1 and display:
            plt.imshow(sgram, cmap='gray', vmin=0, vmax=1)
            title_str = 'Sinogram: {:.2f} views.'
            plt.suptitle(title_str.format(views))
            plt.title('Close to continue.')
          
            # Show the image before all samples have been generated       
            plt.show()
        
        fileName = os.path.join(destDir, 'im_' + str(idx).zfill(4) + '.pth')
        
        torch.save(fbp, fileName)
        
def makeDataset(args, seed=0):
    """This function generates a simulated dataset of grayscale ellipse images.
    The number of ellipses, image, size, etc. can all be set through
    command-line arguments.
    
    Parameters:
        args: A Namespace object containing relevant options and arguments for
            execution.  This has been parsed from sys.argv using the function
            getArgs(). For a list of available options and arguments, see the
            parser.add_argument() calls bin getArgs, or execute this file with
            the -h or --help options.
        seed: (Optional) An integer used to seed random number generators for 
            reproducibility. Defaults to 0.
        
    Returns:
        Nothing
        
    Outputs:
        output_dir: In the directory specified by args.output_dir, this creates
            grayscale images of randomly drawn ellipses.  Images are saved as
            floating point numpy arrays with pixel values in the range (0,1).
            Each image will be numbered as im_xx0.npy, etc. If a directory of
            the correct name does not already exist, it will be created.
            Existing images in the directory will be overwritten.
    """
    # Set seeds for reproducibility
    random.seed(seed)
    np.random.seed(seed)
    
    # Calculate the number of digits to use for file names
    order_samps = int(np.log10(args.samples)) + 1
    
    # For each image that needs generating
    for samp in range(1, args.samples+1):
        image = np.zeros((args.res,args.res), dtype=np.single)
        
        # Randomly pick a number of ellipses from the specified range
        for ips in range(np.random.randint(args.lower_bound,args.upper_bound)):
            ellipseFits = False
            
            # Keep trying random ellipses until one fits in the image
            while not ellipseFits:
                h = np.random.randint(-(args.res/2 - 8),(args.res/2 -8))
                k = np.random.randint(-(args.res/2 -8),(args.res/2 -8))
                a = np.random.randint(args.res/32,args.res/8)
                b = np.random.randint(args.res/32,args.res/8)
                theta = np.random.randint(0,360)
                val = np.random.uniform(low=-0.5, high=0.5, size=None)
                ellipseFits, newEllipse = genEllipse(m=args.res, h=h, k=k, a=a,
                                                     b=b, theta=theta, val=val)
                
            image = np.add(image, newEllipse)
        
        # Shift the image so that zero-valued pixels are now = 0.5
        image = np.add(image, 0.5*np.ones((args.res,args.res),dtype=np.single))
        
        # Clip the image to lie between 0 and 1 for the torch_radon library
        image = np.clip(image, 0.0, 1.0)
            
        # Plot the first image and some of its statistics for verification
        if samp == 1 and args.display:
            plt.imshow(image, cmap='gray', vmin=0, vmax=1)
            title_str = 'Sample Image. Max Intensity: {:.2f}. '
            title_str += 'Min Intensity: {:.2f}. '
            title_str += 'Mode Intensity: {:.2f}.'
            plt.suptitle(title_str.format(np.max(image), np.min(image),
                                       mode(image, axis=None).mode[0]))
            plt.title('Close to continue.')
          
            # Show the image before all samples have been generated       
            plt.show()

        # If necessary, create output directory
        if not os.path.isdir(args.output_dir):
            os.mkdir(args.output_dir)
            
        # Apply Radon transform to this image
        applyRadon(image, samp, args.output_dir, n_views=[1000,143,50],
                   display=args.display)

        
# If this file is run, parse the command-line arguments and the call main()
if __name__ == '__main__':
    args = getArgs(argv)
    makeDataset(args)
    
