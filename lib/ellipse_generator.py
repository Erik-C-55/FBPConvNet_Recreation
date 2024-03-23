"""ellipse_generator.py

Contains helper functions to generate ellipses and verify that they fit in the
images."""

# Standard Imports
import typing

# Third-Party Imports
import numpy as np
from skimage.transform import rotate


def check_ellipse_bounds(m: int, center: typing.Tuple[int, int], a: int,
                         b: int, theta: int) -> bool:
    """Calculates whether an ellipse fits within the original & rotated images.

    Parameters
    ----------
    m : int
        width/height (in pixels) of the (square) image. The origin of
        the graph will be placed at m//2, so the graph bounds are -m//2
        and m//2
    center : (int, int)
        (x, y) coordinates for the center of the ellipse
    a : int
        half the length of the horizontal axis of the ellipse (before rotation)
    b : int
        half the length of the vertical axis of the ellipse (before rotation)
    theta : int
        the angle of ellipse rotation (in degrees) about the origin. Rotation
        is applied before horizontal/vertical shifts.

    Returns
    ----------
    ellipse_fits : bool
        True if the ellipse fits. False if it does not fit.
    """

    # Find original bounding box, centered @ (0,0) (shift by h, k occurs later)
    bbox = np.array([[-a, a, a, -a],
                     [-b, -b, b, b]])

    # Use a rotation matrix to generate the new bounding box
    rot_mat = np.array(
        [[np.cos(np.radians(-theta)), -1*np.sin(np.radians(-theta))],
         [np.sin(np.radians(-theta)), np.cos(np.radians(-theta))]])

    rot_bbox = np.matmul(rot_mat, bbox)

    # Now shift by h, k and shift so the origin is in the image center.
    # Discretize by converting to an int

    shift_array = np.array(
        [[center[0]+m//2, center[0]+m//2, center[0]+m//2, center[0]+m//2],
         [center[1]+m//2, center[1]+m//2, center[1]+m//2, center[1]+m//2]])

    bbox = np.add(bbox, shift_array).astype(np.int32)
    rot_bbox = np.add(rot_bbox, shift_array).astype(np.int32)

    min_x = min(np.min(bbox[0, :]), np.min(rot_bbox[0, :]))
    min_y = min(np.min(bbox[1, :]), np.min(rot_bbox[1, :]))
    max_x = max(np.max(bbox[0, :]), np.max(rot_bbox[0, :]))
    max_y = max(np.max(bbox[1, :]), np.max(rot_bbox[1, :]))

    ellipse_fits = True

    if min_x < 0 or min_y < 0 or max_x > m or max_y > m:
        ellipse_fits = False

    return ellipse_fits


def gen_ellipse(m: int, center: typing.Tuple[int, int],
                a: int = 1, b: int = 1, theta: int = 0,
                val: int = 1) -> typing.Tuple[bool, np.ndarray]:
    """Draws an ellipse in an empty image. Returns an empty image if the
    ellipse does not fit in the given image size.

    Parameters
    ----------
    m : int
        width/height (in pixels) of the (square) image. The origin of
        the graph will be placed at m//2, so the graph bounds are -m//2
        and m//2
    center : (int, int)
        (x, y) coordinates for the center of the ellipse
    a : int, default: 1
        half the length of the horizontal axis of the ellipse (before rotation)
    b : int, default: 1
        half the length of the vertical axis of the ellipse (before rotation)
    theta : int, default: 0
        the angle of ellipse rotation (in degrees) about the origin. Rotation
        is applied before horizontal/vertical shifts.
    val : int, default: 1
        val is the pixel intensity within the ellipse.  All other
        pixels are set to 0.

    Returns
    ----------
    ellipseFits : bool
        True if the ellipse fits. False if it does not fit.
    im : numpy.ndarray
        An int32 array with the ellipse drawn in it. If the specified ellipse
        does not fit in the given image size, the returned image is empty.
    """

    # Check that this ellipse will completely fit in its place before rotating
    ellipse_fits = check_ellipse_bounds(m, (center[0], center[1]), a, b, theta)

    # Start with a blank image
    im = np.zeros((m, m), dtype=np.int32)

    if ellipse_fits:
        # Establish coordinate grid
        x = np.linspace(-m//2, m//2, m)
        y = np.linspace(-m//2, m//2, m)[:, None]  # As a column vector

        x_grid, y_grid = np.meshgrid(x, y)

        # Calculate the ellipse values
        ellipse = ((x_grid-center[0])/a)**2 + ((y_grid-center[1])/b)**2

        # Add the ellipse to the existing image
        im[ellipse <= 1.0] = True

        # Generate a rotated image
        im = rotate(im, theta, center=(center[0]+m//2, center[1]+m//2),
                    preserve_range=True)

        # Convert rotated image to float, scale ellipse by value
        im = im.astype(np.single) * val

    return ellipse_fits, im
