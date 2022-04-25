import numpy as np
from skimage.transform import rotate

def checkEllipseBounds(m,h,k,a,b,theta):
    """This function calculates the bounding box for a rotated ellipse, then
    returns whether or not the ellipse (approximated by its bounding box) fits
    within the original image and the rotated image.
    
    Parameters:
        m (int): width/height (in pixels) of the (square) image. The origin of 
            the graph will be placed at m//2, so the graph bounds are -m//2 
            and m//2
        h (int): x-coordinate for the center of the ellipse
        k (int): y-coordinate for the center of the ellipse
        a (int): a is half the length of the horizontal axis of the ellipse
            (before rotation)
        b (int): b is half the length of the vertical axis of the ellipse
            (before rotation)
        theta (int): theta is the angle of ellipse rotation (in degrees) about
            the origin.  Rotation is applied before horizontal/vertical shifts.
    
    Returns:
        ellipseFits (bool): True if the ellipse fits. False if it does not fit.
    """
    
    # Find original bounding box, centered @ (0,0) (shift by h, k occurs later)
    bbox = np.array([[-a, a, a, -a],
                     [-b, -b, b, b]])
    
    # Use a rotation matrix to generate the new bounding box
    rad = np.radians(-theta)
    c, s = np.cos(rad), np.sin(rad)
    rot_mat = np.array([[c, -1*s],
                        [s, c]])
    
    rot_bbox = np.matmul(rot_mat, bbox)
    
    # Now shift by h, k and shift so the origin is in the image center.
    # Discretize by converting to an int
    
    shift_array = np.array([[h+m//2, h+m//2, h+m//2, h+m//2],
                                [k+m//2, k+m//2, k+m//2, k+m//2]])
    
    bbox = np.add(bbox,shift_array).astype(np.int32)
    rot_bbox = np.add(rot_bbox,shift_array).astype(np.int32)
    
    min_x = min(np.min(bbox[0,:]),np.min(rot_bbox[0,:]))
    min_y = min(np.min(bbox[1,:]),np.min(rot_bbox[1,:]))
    max_x = max(np.max(bbox[0,:]),np.max(rot_bbox[0,:]))
    max_y = max(np.max(bbox[1,:]),np.max(rot_bbox[1,:]))
    
    ellipseFits = True
    
    if min_x < 0 or min_y < 0 or max_x > m or max_y > m:
        ellipseFits = False
        
    return ellipseFits
    
def genEllipse(m=64, h=0, k=0, a=1, b=1, theta=0, val=1):
    """This function draws an ellipse in an empty image. If the specified
    ellipse does not fit in the given image size, the returned image is empty.
    
    Parameters:
        m (int): width/height (in pixels) of the (square) image. The origin of 
            the graph will be placed at m//2, so the graph bounds are -m//2 
            and m//2
        h (int): x-coordinate for the center of the ellipse
        k (int): y-coordinate for the center of the ellipse
        a (int): a is half the length of the horizontal axis of the ellipse
            (before rotation)
        b (int): b is half the length of the vertical axis of the ellipse
            (before rotation)
        theta (int): theta is the angle of ellipse rotation (in degrees) about
            the origin.  Rotation is applied before horizontal/vertical shifts.
        val (int): val is the pixel intensity within the ellipse.  All other 
            pixels are set to 0.
    
    Returns:
        ellipseFits (bool): True if the ellipse fits. False if it does not fit.
        im (ndarray): An int32 array with the ellipse drawn in it. If the 
            specified ellipse does not fit in the given image size, the
            returned image is empty.
            
    """
    
    # Check that this ellipse will completely fit in its place before rotating
    ellipseFits = checkEllipseBounds(m, h, k, a, b, theta)
    
    # Start with a blank image
    im = np.zeros((m,m), dtype=np.int32)
    
    if ellipseFits:
        # Establish coordinate grid
        x = np.linspace(-m//2, m//2, m)
        y = np.linspace(-m//2, m//2, m)[:,None] # As a column vector
        
        x_grid, y_grid = np.meshgrid(x,y)
        
        # Calculate the ellipse values
        ellipse = ((x_grid-h)/a)**2 + ((y_grid-k)/b)**2
        
        # Add the ellipse to the existing image
        im[ellipse <= 1.0] = True
    
        # Generate a rotated image      
        im = rotate(im, theta, center=(h+m//2, k+m//2), preserve_range=True)
        
        # Convert rotated image to float, scale ellipse by value
        im = im.astype(np.single) * val
        
    return ellipseFits, im
