"""genDataset.py

Generates a dataset of random ellipses according to command-line arguments. For
help, run `python genDataset.py --help` in the command-line"""

# Standard Imports
import os
import random
import typing
import argparse
from sys import argv

# Third-Party Imports
import numpy as np
import matplotlib.pyplot as plt
from skimage.transform import radon, iradon

# Import functions from this repo
from ellipseGenerator import genEllipse


def getArgs(cl_args: typing.List[str]) -> argparse.Namespace:
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
    parser.add_argument("filename",type=str,help=argparse.SUPPRESS)
    parser.add_argument('output_dir', type=str,
                        help='directory for holding dataset images')
    
    # Add optional arguments
    parser.add_argument('-d','--display', action='store_true',
                        help='display samples of the first image generated.')
    parser.add_argument('-l','--lower_bound', type=int, default=5,
                        help='a random number of ellipses will be generated.' +
                        ' This is the lower bound for that random number.')
    parser.add_argument('-m','--min_sample', type=int, default =1,
                        help='(integer) the sample to start generating from.' +
                        ' If some samples have already been generated and' +
                        ' the dataset size needs to be increased, setting ' +
                        ' this to a value 1 greater than the number of' +
                        ' samples already generated allows you to "pick up' +
                        ' where you left off"')
    parser.add_argument('-r','--res', type=int, default=512,
                        help='image resolution in pixels (integer).  Image' +
                        ' will be square (res x res).  Res should be a' +
                        ' multiple of 16.')
    parser.add_argument('-s','--samples', type=int, default=10,
                        help='integer number of sample images to generate.')
    parser.add_argument('-u','--upper_bound', type=int, default=15,
                        help='a random number (integer) of ellipses will be' +
                        ' generated.  This is the upper bound for that' +
                        ' random number.')
    
    args = parser.parse_args(cl_args)
    
    # Ensure that the upper bound is greater than the lower bound
    if args.lower_bound > args.upper_bound:
        args.upper_bound = args.lower_bound + 10
    
    return args


def applyRadon(im: np.ndarray, idx: int, directory: str, ord_samps: int,
               display: bool=False) -> None:
    """
    Generate and save 1000-view and low-view reconstructed images

    Parameters
    ----------
    im : np.ndarray
        The image to be transformed and reconstructed
    idx : int
        The index number of the image to be saved
    directory : str
        Filepath for the given number of ellipses
    ord_samps : int
        Number of digits to use for filenames (performs zero-padding)
    display : bool, optional
        Whether to display the generated images. The default is False.
    """
    
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
    sgram_full = radon(im, theta=angles, circle=False, preserve_range=True)
    
    fbp_1000 = iradon(sgram_full, filter_name='ramp', interpolation='linear',
                      circle=False, preserve_range=True)
    
    # Calculate low-view FBP
    angles = np.linspace(0.0, 180.0, 143, endpoint=False)
    sgram = radon(im, theta=angles, circle=False, preserve_range=True)
    fbp_143 = iradon(sgram, filter_name='ramp', interpolation='linear',
                     circle=False, preserve_range=True)
    
    angles = np.linspace(0.0, 180.0, 50, endpoint=False)
    sgram = radon(im, theta=angles, circle=False, preserve_range=True)
    fbp_50 = iradon(sgram, filter_name='ramp', interpolation='linear',
                    circle=False, preserve_range=True)
    
    # Save files
    imName = 'im_'+str(idx).zfill(ord_samps)
    np.save(os.path.join(dirs[0], imName), fbp_1000)
    np.save(os.path.join(dirs[1], imName), fbp_143)
    np.save(os.path.join(dirs[2], imName), fbp_50)
    np.save(os.path.join(dirsgram, imName), sgram_full)
    
    if display and idx==1:
        print('1000-view sinogram size: ' + str(np.shape(sgram)))
        print('143-view sinogram size: ' + str(np.shape(sgram[:,::7])))
        print('50-view sinogram size: ' + str(np.shape(sgram[:,::20])))
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2,2, figsize=(8,8))
        
        ax1.set_title("Original")
        ax1.imshow(im, cmap=plt.cm.Greys_r)
        ax1.axis('off')
        
        ax2.set_title("FBP 1000 Views")
        ax2.imshow(fbp_1000, cmap=plt.cm.Greys_r)
        ax2.axis('off')
        
        ax3.set_title("FBP 143 Views")
        ax3.imshow(fbp_143, cmap=plt.cm.Greys_r)
        ax3.axis('off')
        
        ax4.set_title("FBP 50 Views")
        ax4.imshow(fbp_50, cmap=plt.cm.Greys_r)
        ax4.axis('off')
        
        fig.tight_layout()
        plt.show()
     
        
def makeDataset(args: argparse.Namespace) -> None:
    """Generate a simulated dataset of grayscale ellipse images.
    
    Parameters
    ----------
    args : argparse.Namespace
        Relevant options and arguments for execution.
        
    Outputs
    -------
    output_dir: 
        Grayscale images (np arrays) of randomly drawn ellipses.
    """
       
    # Calculate the number of digits to use for file names
    order_samps = int(np.log10(args.samples)) + 1
    
    # If necessary, create output directory & subfolder
    subfolder = os.path.join(args.output_dir, str(args.lower_bound) + '_' + \
                             str(args.upper_bound - 1) + '_Images')
        
    if not os.path.isdir(args.output_dir):
        os.mkdir(args.output_dir)
    
    if not os.path.isdir(subfolder):
        os.mkdir(subfolder)
    
    # For each image that needs generating
    for samp in range(args.min_sample, args.samples+1):

        # Set seeds for reproducibility. Normally, seeding every iteration is 
        # wasteful.  However, seeding inside the loop allows you to increase
        # you dataset size later without needing to re-generate samples
        # you already have.
        random.seed(samp)
        np.random.seed(samp)

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
                val = np.random.uniform(low=-500, high=500, size=None)
                ellipseFits, newEllipse = genEllipse(m=args.res, h=h, k=k, a=a,
                                                     b=b, theta=theta, val=val)
                
            image = np.add(image, newEllipse)
        
        # Clip the image to lie between -500 and 500
        image = np.clip(image, -500, 500)
            
        # Apply Radon transform to this image
        applyRadon(image, samp, subfolder, order_samps, display=args.display)


# If this file is run, parse the command-line arguments and the call main()
if __name__ == '__main__':
    args = getArgs(argv)
    makeDataset(args)
    
