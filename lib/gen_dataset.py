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
from lib.ellipse_generator import gen_ellipse

# Repo examines 1000-, 143-, & 50-view reconstructions
RECON_VIEWS = (1000, 143, 50)


def get_args(cl_args: typing.List[str]) -> argparse.Namespace:
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
    parser.add_argument('-d', '--display', action='store_true',
                        help='display samples of the first image generated.')
    parser.add_argument(
        '-l', '--lower_bound', type=int, default=5,
        help='Lower bound for the random number of ellipses to be generated.')
    parser.add_argument(
        '-m', '--min_sample', type=int, default=1,
        help='the sample to start generating from. If some samples have ' +
        'already been generated and the dataset size needs to be increased, ' +
        'setting this to a value 1 greater than the number of samples ' +
        'already generated allows you to "pick up where you left off"')
    parser.add_argument(
        '-r', '--res', type=int, default=512, choices=np.arange(64, 1024, 64),
        help='image resolution in pixels. Image  will be square (res x res).')
    parser.add_argument('-s', '--samples', type=int, default=10,
                        help='integer number of sample images to generate.')
    parser.add_argument(
        '-u', '--upper_bound', type=int, default=15,
        help='Upper bound for the random number of ellipses to be generated.')

    args = parser.parse_args(cl_args)

    # Ensure that the upper bound is greater than the lower bound
    if args.lower_bound > args.upper_bound:
        args.upper_bound = args.lower_bound + 10

    return args


def apply_radon(im: np.ndarray, idx: int, directory: str, ord_samps: int,
                display: bool = False) -> None:
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
    dirs = {str(n_views): os.path.join(
        directory, f'{n_views}_Views') for n_views in RECON_VIEWS}
    dirs['Sinograms'] = os.path.join(directory, 'Sinograms')

    for dir_name in dirs.values():
        if not os.path.isdir(dir_name):
            os.mkdir(dir_name)

    for n_angles in RECON_VIEWS:

        # Calculate all view angles (degrees) for this projection
        angles = np.linspace(0.0, 180.0, n_angles, endpoint=False)

        # Calculate sinogram and filtered backprojection
        sgram = radon(im, theta=angles, circle=False, preserve_range=True)
        fbp = iradon(sgram, filter_name='ramp', interpolation='linear',
                     circle=False, preserve_range=True)

        # Save file
        im_name = f'im_{str(idx).zfill(ord_samps)}'
        np.save(os.path.join(dirs[str(n_angles)], im_name), fbp)

        # Save the "full-view" sinogram
        if n_angles == 1000:
            np.save(os.path.join(dirs['Sinograms'], im_name), sgram)

        if display and idx == 1:

            if n_angles == 1000:
                print(f'1000-view sinogram size: {np.shape(sgram)}')

                # Create the subplots, add original and 1000-View FBP
                fig, ax = plt.subplots(2, 2, figsize=(8, 8))

                ax[0, 0].set_title("Original")
                ax[0, 0].imshow(im, cmap=plt.cm.Greys_r)
                ax[0, 0].axis('off')

                ax[0, 1].set_title(f"FBP {n_angles} Views")
                ax[0, 1].imshow(fbp, cmap=plt.cm.Greys_r)
                ax[0, 1].axis('off')

            elif n_angles == 143:
                print(f'143-view sinogram size: {np.shape(sgram[:, ::7])}')

                ax[1, 0].set_title(f"FBP {n_angles} Views")
                ax[1, 0].imshow(fbp, cmap=plt.cm.Greys_r)
                ax[1, 0].axis('off')

            elif n_angles == 50:
                print(f'50-view sinogram size: {np.shape(sgram[:, ::20])}')

                ax[1, 1].set_title(f"FBP {n_angles} Views")
                ax[1, 1].imshow(fbp, cmap=plt.cm.Greys_r)
                ax[1, 1].axis('off')

                fig.tight_layout()
                plt.show()


def make_dataset(args: argparse.Namespace) -> None:
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
    subfolder = os.path.join(args.output_dir, str(args.lower_bound) + '_' +
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

        image = np.zeros((args.res, args.res), dtype=np.single)

        # Randomly pick a number of ellipses from the specified range
        for _ in range(np.random.randint(args.lower_bound, args.upper_bound)):
            ellipse_fits = False

            # Keep trying random ellipses until one fits in the image
            while not ellipse_fits:
                h = np.random.randint(-(args.res/2 - 8), (args.res/2 - 8))
                k = np.random.randint(-(args.res/2 - 8), (args.res/2 - 8))
                a = np.random.randint(args.res/32, args.res/8)
                b = np.random.randint(args.res/32, args.res/8)
                theta = np.random.randint(0, 360)
                val = np.random.uniform(low=-500, high=500, size=None)
                ellipse_fits, new_ellipse = gen_ellipse(
                    args.res, center=(h, k), a=a, b=b, theta=theta, val=val)

            image = np.add(image, new_ellipse)

        # Clip the image to lie between -500 and 500
        image = np.clip(image, -500, 500)

        # Apply Radon transform to this image
        apply_radon(image, samp, subfolder, order_samps, display=args.display)


# If this file is run, parse the command-line arguments and the call main()
if __name__ == '__main__':
    parsed_args = get_args(argv)
    make_dataset(parsed_args)
