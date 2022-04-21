import random
import numpy as np
import matplotlib.pyplot as plt

# Import functions from this repo
from ellipseGenerator import genEllipse

def main(seed=0):
    # random.seed(seed)
    # np.random.seed(seed)
    
    m = 512
    n = 512
    image = np.zeros((m,n), dtype=np.single)
    
    for ipse in range(15):
        ellipseFits = False
        
        while not ellipseFits:
            h = np.random.randint(-(m/2 - 8),(m/2 -8))
            k = np.random.randint(-(n/2 -8),(n/2 -8))
            a = np.random.randint(8,m/4)
            b = np.random.randint(8,n/4)
            theta = np.random.randint(0,360)
            val = np.random.randint(10,40)
            ellipseFits, newEllipse = genEllipse(m=m, n=n, h=h, k=k, a=a, b=b,
                                                 theta=theta, val=val)
            
        image = np.add(image, newEllipse)
        
        
    lower = np.min(image)
    upper = np.max(image)
    
    image = 255.0 * (image - lower)/(upper - lower + 1e-6)
    
    image = image.astype(np.ubyte)
    
    plt.imshow(image, cmap='gray', vmin=0, vmax=255)
    
    print(np.max(image))
    print(np.min(image))
    
if __name__ == '__main__':
    main()
    
