import os
import random
import argparse
import numpy as np
import matplotlib.pyplot as plt

from sys import argv
from skimage.transform import radon, iradon

# Import functions from this repo
from ellipseGenerator import genEllipse

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

def applyRadon(im, idx, directory, ord_samps, display=False):
     
    # Generate names for directories.  If they don't exist, create them
    dirs = []
    for count, views in enumerate([1000, 143, 50]):
        dirs.append(os.path.join(directory, str(views) + '_Views'))
    
        if not os.path.isdir(dirs[count]):
            os.mkdir(dirs[count])
            
    dirsgram = os.path.join(directory, 'Sinograms')
    
    if not os.path.isdir(dirsgram):
        os.mkdir(dirsgram)
    
    # Calculate all view angles (degrees) for 1000-view projection
    angles = np.linspace(0.0, 180.0, 1000, endpoint=False)
    
    # Calculate 'full-view' sinogram and filtered backprojection
    sgram = radon(im, theta=angles, circle=False, preserve_range=True)
    
    fbp_1000 = iradon(sgram, filter_name='ramp', interpolation='linear',
                      circle=False, preserve_range=True)
    
    # Calculate low-view FBP
    fbp_143 = iradon(sgram[:,::7], filter_name='ramp', interpolation='linear',
                     circle=False, preserve_range=True)
    
    fbp_50 = iradon(sgram[:,::20], filter_name='ramp', interpolation='linear',
                    circle=False, preserve_range=True)
    
    # Save files
    imName = 'im_'+str(idx).zfill(ord_samps)
    np.save(os.path.join(dirs[0], imName), fbp_1000)
    np.save(os.path.join(dirs[1], imName), fbp_143)
    np.save(os.path.join(dirs[2], imName), fbp_50)
    np.save(os.path.join(dirsgram, imName), sgram)
    
    if display and idx==1:
        print('1000-view sinogram size: ' + str(np.shape(sgram)))
        print('143-view sinogram size: ' + str(np.shape(sgram[:,::7])))
        print('50-view sinogram size: ' + str(np.shape(sgram[:,::20])))
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2,2, figsize=(8,8))
        
        ax1.set_title("Original")
        ax1.imshow(im, cmap=plt.cm.Greys_r, vmin=0.0, vmax=1.0)
        
        ax2.set_title("FBP 1000 Views")
        ax2.imshow(fbp_1000, cmap=plt.cm.Greys_r, vmin=0.0, vmax=1.0)
        
        ax3.set_title("FBP 143 Views")
        ax3.imshow(fbp_143, cmap=plt.cm.Greys_r, vmin=0.0, vmax=1.0)
        
        ax4.set_title("FBP 50 Views")
        ax4.imshow(fbp_50, cmap=plt.cm.Greys_r, vmin=0.0, vmax=1.0)
        
        fig.tight_layout()
        plt.show()
        
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
        
        # Clip the image to lie between 0 and 1 for the radon transform library
        image = np.clip(image, 0.0, 1.0)

        # If necessary, create output directory & subfolder
        subfolder = os.path.join(args.output_dir, str(args.lower_bound) + '_' + \
                                 str(args.upper_bound - 1) + '_Images')
            
        if not os.path.isdir(args.output_dir):
            os.mkdir(args.output_dir)
        
        if not os.path.isdir(subfolder):
            os.mkdir(subfolder)
            
        # Apply Radon transform to this image
        applyRadon(image, samp, subfolder, order_samps, display=args.display)

# If this file is run, parse the command-line arguments and the call main()
if __name__ == '__main__':
    args = getArgs(argv)
    makeDataset(args)
    
