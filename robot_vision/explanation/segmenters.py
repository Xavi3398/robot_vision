import os
import numpy as np
import zipfile
from tqdm.auto import tqdm
from scipy import ndimage
from skimage.segmentation import slic

from robot_vision.explanation.utils import painter

class Segmenter(object):

    def __init__(self, img):
        """ Base class for segmenting a image into different non-overlapping
            regions (segments).

        Args:
            image (3d numpy array): image to segment, of shape: [height,
            width, 3].
        """
        self.img = img

    def segment(self):
        pass

    def load_segmentation(self, path, compressed=True):
        """Loads segments from memory.

        Args:
            path (string): path to the segments file
            compressed (bool, optional): whether if the file is compressed
                (.zip) or not (.npy). Defaults to True.

        Returns:
            3d numpy array: loaded segments
        """

        # Extract .npy file from .zip, load and delete the temporal .npy file
        if compressed:
            path_zip = path
            path_npy = path[:-3] + 'npy'
            with zipfile.ZipFile(path_zip, 'r') as file_zip:
                file_zip.extractall(os.path.dirname(path_npy))
            segments = np.load(path_npy)  # Load stored segmentation
            os.remove(path_npy)

        # Load the .npy file
        else:
            path_npy = path
            segments = np.load(path_npy)  # Load stored segmentation

        return segments

    def save_segmentation(self, segments, path, compressed=True):
        """Saves segments to memory.

        Args:
            segments (3d numpy array): segments array to save
            path (string): path of the segments file
            compressed (bool, optional): whether if the file is compressed
                (.zip) or not (.npy). Defaults to True.
        """

        # Save first as .npy file, compress to .zip and remove temporal
        # .npy file
        if compressed:
            path_zip = path
            path_npy = path[:-3] + 'npy'
            np.save(path_npy, segments)
            with zipfile.ZipFile(path_zip, 'w') as file_zip:
                file_zip.write(path_npy, arcname=os.path.basename(path_npy),
                               compress_type=zipfile.ZIP_DEFLATED)
                os.remove(path_npy)

        # Save as .npy file
        else:
            path_npy = path
            np.save(path_npy, segments)

    def plot_segments(self, segments, kind='overlay', show_progress=False):
        """ Creates a image to visualize the segments on the image, in a
            similar way to 'label2rgb', from skimage.color.

        Args:
            segments (2d numpy array, optional): segments to visualize on the
                image.
            kind (str, optional): either 'overlay' (random color for each
                segment) or 'avg' (mean color for each segment). Defaults to
                'overlay'.
            show_progress (bool, optional): Whether to show a progress bar.
                Defaults to False.

        Returns:
            3d numpy array: image with the colored segments over it
        """
        return segments2colors(segments, self.img, kind, show_progress)

class SlicSegmenter(Segmenter):

    def __init__(self, img):
        """ Segmenter for obtaining non-overlapping regions (segments) using
            the SLIC technique.

        Args:
            video (4d numpy array): video to segment, of shape: [n_frames,
                height, width, 3].
        """
        super().__init__(img)

    def segment(self, n_segments=200, compactness=5, spacing=[1, 1]):
        """ Segmentation for obtaining non-overlapping regions (segments)
            using the SLIC technique. The arguments are passed directly to
            the slic function, from skimage.segmentation.

        Args:
            n_segments (int, optional): The (approximate) number of labels in
                the segmented output video. Defaults to 200.
            compactness (int, optional): Balances color proximity and space
                proximity. Higher values give more weight to space proximity,
                making superpixel shapes more cubic. This parameter depends
                strongly on video contrast and on the shapes of objects in the
                video. We recommend exploring possible values on a log scale,
                e.g., 0.01, 0.1, 1, 10, 100, before refining around a chosen
                value. Defaults to 5.
            spacing (list, optional): The voxel spacing along each spatial
                dimension. By default, slic assumes uniform spacing (same
                voxel resolution along each spatial/temporal dimension). This
                parameter controls the weights of the distances along the
                spatial/temporal dimensions during k-means clustering.
                Defaults to [0.2,1,1], decreasing importance of temporal
                dimension.

        Returns:
            3d numpy array: computed segments of shape [n_frames, height,
                width], where each element will indicate the region it
                belongs to.
        """
        return slic(self.img, n_segments=n_segments, compactness=compactness,
                    spacing=spacing, start_label=0)
    
class RiseSegmenter(Segmenter):

    def __init__(self, img):
        """ Segmenter for obtaining non-overlapping regions (segments) in a
            RISE way: a small grid is constructed, which will be upscaled
            using linear interpolation in the perturbation phase.

        Args:
            image (3d numpy array): image to segment, of shape: [ height,
            width, 3].
        """
        super().__init__(img)

    def segment(self, n_seg=[10, 10]):
        """ Segmentation in a RISE way: a small grid is constructed, which
            will be upscaled using linear interpolation in the perturbation
            phase.

        Args:
            n_seg (list, optional): number of regions for the grid along each
                axis. Total of segments will be equal to the product of each
                element of the list. Defaults to [5, 10, 10] (500 segments).

        Returns:
            3d numpy array: computed segments of shape n_seg, where each
                element will indicate the region it belongs to.
        """
        segments = np.zeros(shape=n_seg, dtype=int)
        id_seg = 0
        for y in range(n_seg[0]):
            for x in range(n_seg[1]):
                segments[y, x] = id_seg
                id_seg += 1
        return segments


def load_segmentation(path, compressed=True):
    """Loads segments from memory.

    Args:
        path (string): path to the segments file
        compressed (bool, optional): whether if the file is compressed
            (.zip) or not (.npy). Defaults to True.

    Returns:
        3d numpy array: loaded segments
    """

    # Extract .npy file from .zip, load and delete the temporal .npy file
    if compressed:
        path_zip = path
        path_npy = path[:-3] + 'npy'
        with zipfile.ZipFile(path_zip, 'r') as file_zip:
            file_zip.extractall(os.path.dirname(path_npy))
        segments = np.load(path_npy)  # Load stored segmentation
        os.remove(path_npy)

    # Load the .npy file
    else:
        path_npy = path
        segments = np.load(path_npy)  # Load stored segmentation

    return segments


def save_segmentation(segments, path, compressed=True):
    """Saves segments to memory.

    Args:
        segments (3d numpy array): segments array to save
        path (string): path of the segments file
        compressed (bool, optional): whether if the file is compressed
            (.zip) or not (.npy). Defaults to True.
    """

    # Save first as .npy file, compress to .zip and remove temporal
    # .npy file
    if compressed:
        path_zip = path
        path_npy = path[:-3] + 'npy'
        np.save(path_npy, segments)
        with zipfile.ZipFile(path_zip, 'w') as file_zip:
            file_zip.write(path_npy, arcname=os.path.basename(path_npy),
                           compress_type=zipfile.ZIP_DEFLATED)
            os.remove(path_npy)

    # Save as .npy file
    else:
        path_npy = path
        np.save(path_npy, segments)
    

def segments2colors(segments, img, kind='overlay', show_progress=False):
    """ Shows the segmentation on the input image. Works in the same way as
        label2rgb from skimage.color, either using random or average colors to
        fill the different regions.

    Args:
        segments (3d numpy array): segmentation of the image, of shape:
            [n_frames, height, width], where each element represents the region
            (or segment) a pixel belongs to.
        img (4d numpy array): image that is being segmented.
        kind (str, optional): either 'overlay' to display a random color for
            each region or 'avg' to use the mean color of the region. When
            using 'overlay', the image is shown in the background, merging it
            with the segmentatino colors. Defaults to 'overlay'.
        show_progress (bool, optional): whether to show the progress of
            computing the colored image. Defaults to False.

    Returns:
        _type_: _description_
    """

    id_segments = np.unique(segments)
    colors = np.zeros(shape=segments.shape + (3,), dtype='uint8')
    progress = tqdm(id_segments) if show_progress else id_segments

    # if segments need rescale (RISE segmentation)
    if segments.shape != img.shape[:2]:
        scale = [img.shape[i] / segments.shape[i]
                 for i in range(len(segments.shape))]
        segments_big = ndimage.zoom(segments, scale, order=0)

    for id_seg in progress:
        mask = segments == id_seg

        if kind == 'overlay':
            colors[mask, :] = np.random.randint(0, 255, 3)
        elif kind == 'avg':
            if segments.shape != img.shape[:2]:
                mask_img = segments_big == id_seg
                colors[mask, :] = np.mean(img[mask_img, :], axis=0)
            else:
                colors[mask, :] = np.mean(img[mask, :], axis=0)

    # if segments need rescale (RISE segmentation)
    if segments.shape != img.shape[:2]:
        colors = ndimage.zoom(colors, scale + [1,], order=1)

    if kind == 'overlay':
        return painter(colors, img)
    else:
        return colors