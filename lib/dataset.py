"""dataset.py

Creates the dataset, dataloader for use in training/evaluating the model."""

# Standard Imports
import os
import typing

# Third-Party Imports
import torch
import numpy as np
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import Compose, RandomHorizontalFlip, \
    RandomVerticalFlip

NUM_VIEWS_FULL = 1000  # Num. Views in "full-view" filtered back projection


class FBPDataset(Dataset):
    """A class for the FBPConvNet Datataset

    Parameters
    ----------
    n_ellipse : (int, int)
        (min, max) # of ellipses per image. Helps find pre-generated image dir
    low_views : int
        Reduced number of views used for low-view filtered back projection
    transform : torch.nn.Module
        Any torchvision transforms to be applied to this data
    mode : str
        'train', 'validation', or 'test'
    n_samps: int
        Number of samples to use for all the data (train+val+test)
    """

    def __init__(self, n_ellipse: typing.Tuple[int] = (25, 34),
                 low_views: int = 50, transform: torch.nn.Module = None,
                 mode: str = 'train', n_samps: int = 500):
        self.n_ellipse = n_ellipse
        self.low_views = low_views
        self.transform = transform
        self.mode = mode.lower()
        self.n_samps = n_samps

        self._obtain_im_list()

    def _obtain_im_list(self) -> None:
        """Obtains a list of filepaths for dataset images"""

        # Name of directory where images are stored
        im_dir = os.path.join('imageData',
                              f'{self.n_ellipse[0]}_{self.n_ellipse[1]}_Images')

        # Since the images are randomly generated anyway, the training set can
        # simply be the first 90% of images, with the next 5% for validation &
        # the last 5% for test
        train_ims = int(0.9*self.n_samps)
        val_ims = int(np.floor(0.05*self.n_samps))
        test_ims = int(np.ceil(0.05*self.n_samps))

        # Calculate the ranges for each set
        if self.mode == 'train':
            lower = 1
            upper = train_ims + 1

        elif self.mode == 'validation':
            lower = train_ims + 1
            upper = train_ims + val_ims + 1

        elif self.mode == 'test':
            lower = train_ims + val_ims + 1
            upper = train_ims + val_ims + test_ims + 1

        self.full_views_list = [os.path.join(im_dir, f'{NUM_VIEWS_FULL}_Views',
                                             f'im_{im:03}.npy')
                                for im in range(lower, upper)]
        self.low_views_list = [os.path.join(im_dir, f'{self.low_views}_Views',
                                            f'im_{im:03}.npy')
                               for im in range(lower, upper)]

        for im in range(lower, upper):
            im_name = 'im_' + str(im).zfill(4) + '.npy'
            self.full_views_list.append(os.path.join(
                im_dir, f'{NUM_VIEWS_FULL}_Views', im_name))
            self.low_views_list.append(os.path.join(
                im_dir, f'{self.low_views}_Views', im_name))

    def __getitem__(self, index) -> typing.Tuple[torch.Tensor, torch.Tensor]:
        """Loads a single input, ground truth pair from the dataset.

        Parameters
        ----------
        index : int
            Index of the dataset element to load

        Returns
        ----------
        torch.Tensor
            The low-view filtered backprojection image (model input)
        torch.Tensor
            The "full-view"" filtered backprojection image (ground truth)
        """

        # Load ground truth and input as tensors
        full_fbp = torch.from_numpy(np.load(self.full_views_list[index]))
        low_fbp = torch.from_numpy(np.load(self.low_views_list[index]))

        # Expand tensors to preserve channel dimension.
        full_fbp = full_fbp.unsqueeze(dim=0)
        low_fbp = low_fbp.unsqueeze(dim=0)

        # Transform ground truth and input. To ensure ground truth and low-view
        # images receive the same transformation, first combine them along a
        # 'frames' dimension to give a tensor of dimensions
        # [Frames =2, Channels=1, Height, Width]

        if self.transform is not None:
            transformed = self.transform(torch.stack((full_fbp, low_fbp),
                                                     dim=0))
            full_fbp = transformed[0, :, :, :]
            low_fbp = transformed[1, :, :, :]

        return low_fbp, full_fbp

    def __len__(self) -> int:
        """Returns dataset length

        Returns
        ----------
        int
            Dataset length
        """

        return len(self.full_views_list)


if __name__ == '__main__':

    augmentations = Compose([RandomHorizontalFlip(), RandomVerticalFlip()])

    imLoader = DataLoader(FBPDataset(n_ellipse=(25, 34),
                                     transform=augmentations),
                          batch_size=2, shuffle=False, num_workers=1)

    low_view, full_view = iter(imLoader).next()

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(8, 8))

    ax1.imshow(low_view[0, :, :, :].squeeze(0), cmap=plt.cm.Greys_r)
    ax1.set_title('50-view FBP 1')

    ax2.imshow(low_view[1, :, :, :].squeeze(0), cmap=plt.cm.Greys_r)
    ax2.set_title('50-view FBP 2')

    ax3.imshow(full_view[0, :, :, :].squeeze(0), cmap=plt.cm.Greys_r)
    ax3.set_title('1000-view FBP 1')

    ax4.imshow(full_view[1, :, :, :].squeeze(0), cmap=plt.cm.Greys_r)
    ax4.set_title('1000-view FBP 2')

    fig.tight_layout()
    plt.show()
