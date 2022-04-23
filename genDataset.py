import os
import random
import numpy as np
import matplotlib.pyplot as plt

from sys import argv

from scipy.stats import mode

# Import functions from this repo
from ellipseGenerator import genEllipse
from userInput import getArgs

def main(args, seed=0):
    random.seed(seed)
    np.random.seed(seed)
    
    order_samps = int(np.log10(args.samples)) + 1
    
    for samp in range(1, args.samples+1):
        image = np.zeros((args.res,args.res), dtype=np.single)
        
        for ips in range(np.random.randint(args.lower_bound,args.upper_bound)):
            ellipseFits = False
            
            # Keep trying ellipses until one fits
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
  
if __name__ == '__main__':
    args = getArgs(argv)
    main(args)
    
