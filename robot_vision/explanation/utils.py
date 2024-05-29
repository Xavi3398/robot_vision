import os
import cv2
import numpy as np

def load_img(path):
    """ Load a image from path.

    Args:
        path (string): path to the image.

    Returns:
        img (3d numpy array): loaded image as a numpy array of shape:
            [n_frames, height, width, channels]
    """
    return cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)


def save_img(img, path, img_name=None, bgr=False):
    """ Save a img to a path.

    Args:
        img (3d numpy array): img to save
        path (string): path to the folder where the image should be stored
            if img_name is not None. Otherwise, the path should already
            contain the img_name.
        img_name (string, optional): name of the image to store. Defaults to
            None.
        bgr (bool, optional): Whether the image is already in BGR format or
            not. Defaults to False.
    """
    cv2.imwrite(
        path if img_name is None else os.path.join(path, img_name),
        cv2.cvtColor(img, cv2.COLOR_BGR2RGB) if bgr else img)

def painter(img1, img2, alpha2=0.5):
    """ Merges two images into one, according to alpha factor.

    Args:
        img1: 1st image
        img2: 2nd image
        alpha2: Importance of 2nd image. 1 maximum, 0 minimum.

    Returns:
        Merged images.
    """
    return (img1.astype('float') * (1 - alpha2)
            + img2.astype('float') * alpha2).astype('uint8')


def histogram_stretching(img, h_min=0, h_max=1):
    max_value = np.max(img)
    min_value = np.min(img)
    if max_value > 0 and min_value != max_value:
        return h_min+(h_max-h_min)*(img-min_value)/(max_value-min_value)
    else:
        return img