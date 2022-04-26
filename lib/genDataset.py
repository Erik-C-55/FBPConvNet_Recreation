import os
import random
import argparse
import numpy as np
import matplotlib.pyplot as plt

from sys import argv
from scipy.stats import mode

# Import functions from this repo
from ellipseGenerator import genEllipse

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
    parser.add_argument('-d','--display', action='store_true')
    parser.add_argument('-l','--lower_bound', type=int, default=5,
                        help='a random number (integer) of ellipses will be generated.  This is the lower bound for that random number.')
    parser.add_argument('-r','--res', type=int, default=512,
                        help='image resolution in pixels.  Image will be square (res x res).  Resolution should be a multiple of 16.')
    parser.add_argument('-s','--samples', type=int, default=10,
                        help='number of sample images to generate')
    parser.add_argument('-u','--upper_bound', type=int, default=15,
                        help='a random number (integer) of ellipses will be generated.  This is the upper bound for that random number.')
    
    args = parser.parse_args(CLArgs)
    
    # Ensure that the upper bound is greater than the lower bound
    if args.lower > args.upper:
        args.upper = args.lower + 10
    
    return args

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
    
        # Save the image to the specified output directory
        if not os.path.isdir(args.output_dir):
            os.mkdir(args.output_dir)
            
        np.save(os.path.join(args.output_dir,
                             'im_'+str(samp).zfill(order_samps)), image)
            
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

# If this file is run, parse the command-line arguments and the call main()
if __name__ == '__main__':
    args = getArgs(argv)
    makeDataset(args)
    
