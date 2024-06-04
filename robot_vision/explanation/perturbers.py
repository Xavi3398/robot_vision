import os
import numpy as np
from tqdm.auto import tqdm
import copy
from sklearn.utils import check_random_state
from scipy import ndimage
from functools import partial
import shap

import cv2

from robot_vision.explanation.utils import load_img, save_img

class Perturber(object):

    def __init__(self, img, segments, classifier_fn, hide_color=None,
                 random_state=None):
        """ Base class for creating perturbed versions of a image by occluding
            in different ways the regions stablished by a segmentation, and
            obtaining a prediction for each one using the passed function.

        Args:
            image (3d numpy array): image to perturb, of shape:  [height,
            width, 3].
            segments (2d numpy array): segmentation of the image, of shape:
                [height, width], where each element indicates the
                region it belongs to.
            classifier_fn (_type_): function to predict the class of a image.
            hide_color (int, optional): color to use occlude a region. If None,
                the mean color of a region thrughout the image is used for that
                region. Defaults to None.
            random_state (int, optional): seed to use for replicating
                perturbations. Defaults to None.
        """
        self.img = img
        self.segments = segments
        self.n_features = np.unique(self.segments).shape[0]
        self.classifier_fn = classifier_fn
        self.random_state = check_random_state(random_state)

        # Init fudged image
        self.fudged_img = img.copy()

        # Use mean color for each region
        if hide_color is None:

            # For RISE-like perturbations: the segmentation array is smaller
            # and has to be rescaled. Color of pixels between regions will be
            # the result of interpolation between mean colors of regions.
            if segments.shape != img.shape[:2]:
                scale = [img.shape[i] / segments.shape[i]
                         for i in range(len(segments.shape))]
                segments_big = ndimage.zoom(segments, scale, order=0)
                fudged_small = np.zeros(shape=segments.shape+(3,),
                                        dtype=img.dtype)
                for x in np.unique(segments):
                    fudged_small[segments == x] = (
                        np.mean(img[segments_big == x][:, 0]),
                        np.mean(img[segments_big == x][:, 1]),
                        np.mean(img[segments_big == x][:, 2]))
                self.fudged_img = ndimage.zoom(fudged_small, scale + [1,],
                                                 order=1)

            # Each region will have the same color: the mean color of
            # that region
            else:
                for x in np.unique(segments):
                    self.fudged_img[segments == x] = (
                        np.mean(img[segments == x][:, 0]),
                        np.mean(img[segments == x][:, 1]),
                        np.mean(img[segments == x][:, 2]))

        # Use a specified color for all regions
        else:
            self.fudged_img[:] = hide_color

        self.data_fn = self.get_data

    def get_data(self, num_samples):
        pass

    def get_non_perturbed_samples(self, num_samples):
        pass

    def perturb_and_predict(self, data, batch_size=10, progress_bar=True,
                            save_imgs_path=None, load_imgs_path=None,
                            dont_predict=False):
        """ Generates perturbed images and their predictions. It has been
            adapted from the public LIME repository.

        Args:
            data (2d numpy array): array of shape: [num_samples, num_features]
                with 0 where a region is occluded for a specific instance, and
                1 where not.
            batch_size (int, optional): number of images to be passed together
                to the prediction function. Defaults to 10.
            progress_bar (bool, optional): Whether to display a progress bar or
                not. Defaults to True.
            save_imgs_path (string, optional): path of a folder where
                perturbed instances should be stored. No images will be stored
                if None. Defaults to None.
            load_imgs_path (string, optional): path of the folder from where
                to load saved perturbed instances. Defaults to None.
            dont_predict (bool, optional): whether to call classifier_fn or
                not. Useful if only perturbations are wanted, or want to be
                stored. Defaults to False.

        Returns:
            A numpy array of shape: [num_samples, num_classes], conaining the
            prediction of classifier_fn for each class and perturbed sample.
        """

        self.data = data
        labels = []
        imgs = []
        rows = tqdm(data) if progress_bar else data

        # Compute first the mask for each region
        region_mask = np.zeros(shape=(self.n_features,) + self.segments.shape,
                               dtype=bool)
        for i in range(self.n_features):
            region_mask[i, ...] = self.segments == i

        for n_sample, row in enumerate(rows):

            # Generate perturbed img
            if load_imgs_path is None:

                # Initialize perturbed img as a simple copy
                temp = copy.deepcopy(self.img)

                # Get all segments that should be occluded
                zeros = np.where(row == 0)[0]

                # If at least one region should be occluded
                if len(zeros > 0):

                    # Join masks of all regions that should be occluded using
                    # logical OR
                    mask = region_mask[zeros[0], ...]
                    for j in zeros[1:]:
                        mask = np.logical_or(mask, region_mask[j, ...])

                    # If mask shape is smaller, a rescale is needed. Used for
                    # RISE-like perturbations. The mask will be of floats, using
                    # interpolation between occluded and not occluded regions.
                    if mask.shape != temp.shape[:2]:
                        scale = [temp.shape[i] / mask.shape[i]
                                for i in range(len(mask.shape))]
                        mask = ndimage.zoom(np.logical_not(mask).astype(float),
                                            scale, order=1)
                        inv_mask = 1 - mask
                        temp[..., 0] = \
                            (temp[..., 0]*mask+inv_mask*self.fudged_img[..., 0]
                            ).astype('uint8')
                        temp[..., 1] = \
                            (temp[..., 1]*mask + inv_mask*self.fudged_img[..., 1]
                            ).astype('uint8')
                        temp[..., 2] = \
                            (temp[..., 2]*mask + inv_mask*self.fudged_img[..., 2]
                            ).astype('uint8')

                    # Binary mask (LIME-like), where all occluded regions are set
                    # to the color of that regions in the fudged img
                    else:
                        temp[mask, :] = self.fudged_img[mask, :]

                # Save perturbed image if specified
                if save_imgs_path is not None:
                    save_img(temp, save_imgs_path, str(n_sample)+'.png')

            # Load perturbed img from memory
            else:
                temp = load_img(os.path.join(load_imgs_path,
                                               str(n_sample)+'.png'))

            # Add to the list of imgs to predict if not dont_predict
            if not dont_predict:
                imgs.append(temp)

            # Predict all imgs in list when the batch_size is full. Append
            # results to list
            if len(imgs) == batch_size and not dont_predict:
                preds = self.classifier_fn(np.array(imgs))
                labels.extend(preds)
                imgs = []

        # Predict remaining imgs in list. Append results to list
        if len(imgs) > 0 and not dont_predict:
            preds = self.classifier_fn(np.array(imgs))
            labels.extend(preds)

        self.labels = np.array(labels)

        return np.array(labels)

class MultiplePerturber(Perturber):

    def __init__(self, img, segments, classifier_fn, hide_color=None,
                 random_state=None):
        """ Perturber which occludes multiple regions for each perturbed image
            (approximately half of the regions). The number of perturbed images
            can be chosen with num_samples.

            The original image without occlusions is appended to the start.

        Args:
            image (3d numpy array): image to perturb, of shape:  [height,
            width, 3].
            segments (2d numpy array): segmentation of the image, of shape:
                [height, width], where each element indicates the region it
                belongs to.
            classifier_fn (_type_): function to predict the class of a image
            hide_color (int, optional): color to use occlude a region. If None,
                the mean color of a region thrughout the image is used for that
                region. Defaults to None.
            random_state (int, optional): seed to use for replicating
                perturbations. Defaults to None.
        """
        super().__init__(img, segments, classifier_fn, hide_color,
                         random_state)

    def perturb(self, num_samples=500, exact=False, p=0.5, batch_size=10,
                progress_bar=True, save_imgs_path=None,
                load_imgs_path=None, dont_predict=False):
        """ Generates images with multiple perturbations and their predictions.

        Args:
            num_samples (int, optional): number of desired perturbed images.
                Defaults to 500.
            exact (bool, optional): whether to perturb exactly (True) or
                approximately (False) a (1-p) proportion of the regions.
                Defaults to False.
            p (float, optional): probability for each region to be occluded.
                Defaults to 0.5.
            batch_size (int, optional): number of images to be passed together
                to the prediction function. Defaults to 10.
            progress_bar (bool, optional): Whether to display a progress bar or
                not. Defaults to True.
            save_imgs_path (string, optional): path of a folder where
                perturbed instances should be stored. No images will be stored
                if None. Defaults to None.
            load_imgs_path (string, optional): path of the folder from where
                to load saved perturbed instances. Defaults to None.
            dont_predict (bool, optional): whether to call classifier_fn or
                not. Useful if only perturbations are wanted, or want to be
                stored. Defaults to False.

        Returns:
            A numpy array of shape: [num_samples, num_classes], conaining the
            prediction of classifier_fn for each class and perturbed sample.
        """

        # Create data according to perturbation type
        data = self.get_data(num_samples, exact, p)

        # Perturb images accordingly and run inference on them
        return data, self.perturb_and_predict(
            data, batch_size, progress_bar, save_imgs_path,
            load_imgs_path, dont_predict)

    def get_data(self, num_samples=500, exact=False, p=0.5):
        """ Occludes multiple regions for each perturbed image.

        Args:
            num_samples (int, optional): number of desired perturbed images.
                Defaults to 500.
            exact (bool, optional): whether to perturb exactly (True) or
                approximately (False) a (1-p) proportion of the regions.
                Defaults to False.
            p (float, optional): probability for each region to be occluded.
                Defaults to 0.5.

        Returns:
            2d numpy array: array of shape: [num_samples+1, num_features]. This
            array will have 0 where a region is occluded for a specific
            instance, and 1 where not. The first row corresponds to the
            original image without perturbations.
        """

        # Shuffle feature indexes and occlude the first (1 - p) of them
        if exact:
            data = np.zeros(shape=(num_samples, self.n_features), dtype=int)
            for i in range(num_samples):
                regions = np.array(range(self.n_features))
                np.random.shuffle(regions)
                data[i, regions[:int(p*len(regions))]] = 1

        # # Randomly set each region of each sample to 0 or 1
        else:
            data = np.zeros(shape=(num_samples, self.n_features), dtype=int)
            r_data = np.random.random(num_samples * self.n_features) \
                .reshape((num_samples, self.n_features))
            data[r_data >= p] = 0
            data[r_data < p] = 1

        # Add original image at the beginning
        data = np.vstack([np.ones(self.n_features, dtype=int), data])
        return data

    def get_non_perturbed_samples(self, num_samples):
        """ Computes which of the rows in the returned data correspond to
            non-perturbed samples.

        Args:
            num_samples (int): number of desired samples when calling to
                get_data.

        Returns:
            list: list with indexs of the non perturbed samples.
        """
        return [0]
    

class SinglePerturber(Perturber):

    def __init__(self, img, segments, classifier_fn, hide_color=None,
                 random_state=None):
        """ Perturber which occludes only one region for each perturbed img.
            The number of samples will be equal to the number of regions in the
            segmentation.

            Useful for computing the difference between occluding or not
            occluding each of the regions (ablation).

            The original img without occlusions is appended to the start.

        Args:
            img (3d numpy array): img to perturb, of shape:  [height, width, 3].
            segments (2d numpy array): segmentation of the img, of shape:
                [height, width], where each element indicates the
                region it belongs to.
            classifier_fn (_type_): function to predict the class of a img
            hide_color (int, optional): color to use occlude a region. If None,
                the mean color of a region thrughout the img is used for that
                region. Defaults to None.
            random_state (int, optional): seed to use for replicating
                perturbations. Defaults to None.
        """
        super().__init__(img, segments, classifier_fn, hide_color,
                         random_state)

    def perturb(self, batch_size=10, progress_bar=True, save_imgs_path=None,
                load_imgs_path=None, dont_predict=False):
        """ Generates imgs with a single perturbation and their predictions.

        Args:
            batch_size (int, optional): number of imgs to be passed together
                to the prediction function. Defaults to 10.
            progress_bar (bool, optional): Whether to display a progress bar or
                not. Defaults to True.
            save_imgs_path (string, optional): path of a folder where
                perturbed instances should be stored. No imgs will be stored
                if None. Defaults to None.
            load_imgs_path (string, optional): path of the folder from where
                to load saved perturbed instances. Defaults to None.
            dont_predict (bool, optional): whether to call classifier_fn or
                not. Useful if only perturbations are wanted, or want to be
                stored. Defaults to False.

        Returns:
            A numpy array of shape: [num_samples, num_classes], conaining the
            prediction of classifier_fn for each class and perturbed sample.
        """
        # Create data according to perturbation type
        data = self.get_data()

        # Perturb imgs accordingly and run inference on them
        return data, self.perturb_and_predict(
            data, batch_size, progress_bar, save_imgs_path,
            load_imgs_path, dont_predict)

    def get_data(self):
        """ Occludes only one region for each perturbed img. Useful for
            computing the difference between occluding or not occluding each of
            the regions. The number of samples will be equal to the number of
            regions in the segmentation.

        Returns:
            2d numpy array: array of shape: [num_samples+1, num_features]. This
            array will have 0 where a region is occluded for a specific
            instance, and 1 where not. The first row corresponds to the
            original img without perturbations.
        """

        # Use identity matrix with logical not to generate data
        data = np.logical_not(np.eye(self.n_features, self.n_features,
                                     dtype=bool)).astype(int)

        # Add original img at the beginning
        data = np.vstack([np.ones(self.n_features, dtype=int), data])
        return data

    def get_non_perturbed_samples(self, num_samples):
        """ Computes which of the rows in the returned data correspond to
            non-perturbed samples.

        Args:
            num_samples (int): number of desired samples when calling to
                get_data.

        Returns:
            list: list with indexs of the non perturbed samples.
        """
        return [0]


class AllButOnePerturber(Perturber):

    def __init__(self, img, segments, classifier_fn, hide_color=None,
                 random_state=None):
        """ Perturber which occludes all regions but one for each perturbed
            img. The number of samples will be equal to the number of regions
            in the segmentation.

            Useful for computing the difference between occluding or not
            occluding each of the regions.

            The original img without occlusions is appended to the start.

        Args:
            img (3d numpy array): img to perturb, of shape:  [height, width, 3].
            segments (2d numpy array): segmentation of the img, of shape:
                [height, width], where each element indicates the
                region it belongs to.
            classifier_fn (_type_): function to predict the class of a img
            hide_color (int, optional): color to use occlude a region. If None,
                the mean color of a region thrughout the img is used for that
                region. Defaults to None.
            random_state (int, optional): seed to use for replicating
                perturbations. Defaults to None.
        """
        super().__init__(img, segments, classifier_fn, hide_color,
                         random_state)

    def perturb(self, batch_size=10, progress_bar=True, save_imgs_path=None,
                load_imgs_path=None, dont_predict=False):
        """ Generates imgs with a single perturbation and their predictions.

        Args:
            batch_size (int, optional): number of imgs to be passed together
                to the prediction function. Defaults to 10.
            progress_bar (bool, optional): Whether to display a progress bar or
                not. Defaults to True.
            save_imgs_path (string, optional): path of a folder where
                perturbed instances should be stored. No imgs will be stored
                if None. Defaults to None.
            load_imgs_path (string, optional): path of the folder from where
                to load saved perturbed instances. Defaults to None.
            dont_predict (bool, optional): whether to call classifier_fn or
                not. Useful if only perturbations are wanted, or want to be
                stored. Defaults to False.

        Returns:
            A numpy array of shape: [num_samples, num_classes], conaining the
            prediction of classifier_fn for each class and perturbed sample.
        """
        # Create data according to perturbation type
        data = self.get_data()

        # Perturb imgs accordingly and run inference on them
        return data, self.perturb_and_predict(
            batch_size, progress_bar, save_imgs_path,
            load_imgs_path, dont_predict)

    def get_data(self):
        """ Occludes only one region for each perturbed img. Useful for
            computing the difference between occluding or not occluding each of
            the regions. The number of samples will be equal to the number of
            regions in the segmentation.

        Returns:
            2d numpy array: array of shape: [num_samples+1, num_features]. This
            array will have 0 where a region is occluded for a specific
            instance, and 1 where not. The first row corresponds to the
            original img without perturbations.
        """

        # Use identity matrix with to generate data
        data = np.eye(self.n_features, self.n_features, dtype=bool).astype(int)

        # Add fully perturbed img at the beginning
        data = np.vstack([np.zeros(self.n_features, dtype=int), data])
        return data

    def get_non_perturbed_samples(self, num_samples):
        """ Computes which of the rows in the returned data correspond to
            non-perturbed samples.

        Args:
            num_samples (int): number of desired samples when calling to
                get_data.

        Returns:
            list: list with indexs of the non perturbed samples.
        """
        return [0]

    def perturb_and_predict(self, batch_size=10, progress_bar=True,
                            save_imgs_path=None, load_imgs_path=None,
                            dont_predict=False):
        """ Generates perturbed imgs and their predictions. It has been
            adapted from the public LIME repository. Method supersceded
            because it is faster to write only the non perturbed region
            than perturb all regions but one.

        Args:
            batch_size (int, optional): number of imgs to be passed together
                to the prediction function. Defaults to 10.
            progress_bar (bool, optional): Whether to display a progress bar or
                not. Defaults to True.
            save_imgs_path (string, optional): path of a folder where
                perturbed instances should be stored. No imgs will be stored
                if None. Defaults to None.
            load_imgs_path (string, optional): path of the folder from where
                to load saved perturbed instances. Defaults to None.
            dont_predict (bool, optional): whether to call classifier_fn or
                not. Useful if only perturbations are wanted, or want to be
                stored. Defaults to False.
        Returns:
            A numpy array of shape: [num_samples, num_classes], conaining the
            prediction of classifier_fn for each class and perturbed sample.
        """

        labels = []
        imgs = []

        # Progress bar
        if progress_bar:
            progress = tqdm(range(self.n_features+1))

        # Fully perturbed img
        imgs.append(self.fudged_img)

        for n_sample in range(self.n_features):

            # Generate perturbed img
            if load_imgs_path is None:

                # Initialize perturbed img as a simple copy
                temp = copy.deepcopy(self.fudged_img)
                mask = self.segments == n_sample

                # If mask shape is smaller, a rescale is needed. Used for
                # RISE-like perturbations. The mask will be of floats, using
                # interpolation between occluded and not occluded regions.
                if mask.shape != temp.shape[:3]:
                    scale = [temp.shape[i] / mask.shape[i]
                             for i in range(len(mask.shape))]
                    mask = ndimage.zoom(np.logical_not(mask).astype(float),
                                        scale, order=1)
                    inv_mask = 1 - mask
                    temp[..., 0] = \
                        (temp[..., 0]*mask+inv_mask*self.img[..., 0]
                         ).astype('uint8')
                    temp[..., 1] = \
                        (temp[..., 1]*mask + inv_mask*self.img[..., 1]
                         ).astype('uint8')
                    temp[..., 2] = \
                        (temp[..., 2]*mask + inv_mask*self.img[..., 2]
                         ).astype('uint8')

                # Binary mask (LIME-like), where all occluded regions are set
                # to the color of that regions in the fudged img
                else:
                    temp[mask, :] = self.img[mask, :]

                # Save perturbed img if specified
                if save_imgs_path is not None:
                    save_img(temp, save_imgs_path, str(n_sample)+'.png')

            # Load perturbed img from memory
            else:
                temp = load_img(os.path.join(load_imgs_path,
                                               str(n_sample)+'.png'))

            # Add to the list of imgs to predict if not dont_predict
            if not dont_predict:
                imgs.append(temp)

            # Predict all imgs in list when the batch_size is full. Append
            # results to list
            if len(imgs) == batch_size and not dont_predict:
                preds = self.classifier_fn(np.array(imgs))
                labels.extend(preds)
                imgs = []
                progress.update(batch_size)
            elif dont_predict:
                progress.update(1)

        # Predict remaining imgs in list. Append results to list
        if len(imgs) > 0 and not dont_predict:
            preds = self.classifier_fn(np.array(imgs))
            labels.extend(preds)
            progress.update(len(imgs))

        return np.array(labels)


class AccumPerturber(Perturber):

    def __init__(self, img, segments, classifier_fn, hide_color=None,
                 random_state=None):
        """ Perturber which starts with all regions occluded and 'unoccludes'
            one on each iteration, until the original img is obtained. This
            process is repeated num_samples times, so the number of perturbed
            imgs obtained is equal to num_features*num_samples.

            Useful for observing the change in prediction confidence when
            adding the different regions (features) one by one (SHAP).

            The original img without occlusions and the completely occluded
            img are appended to the start.

        Args:
            img (3d numpy array): img to perturb, of shape:  [height, width, 3].
            segments (2d numpy array): segmentation of the img, of shape:
                [height, width], where each element indicates the
                region it belongs to.
            classifier_fn (_type_): function to predict the class of a img
            hide_color (int, optional): color to use occlude a region. If None,
                the mean color of a region thrughout the img is used for that
                region. Defaults to None.
            random_state (int, optional): seed to use for replicating
                perturbations. Defaults to None.
        """
        super().__init__(img, segments, classifier_fn, hide_color,
                         random_state)

    def perturb(self, num_samples=5, batch_size=10, progress_bar=True,
                save_imgs_path=None, load_imgs_path=None,
                dont_predict=False):
        """ Generates imgs with accumulated perturbations and their
            predictions.

        Args:
            num_samples (int, optional): number of desired perturbed imgs.
                Defaults to 500.
            batch_size (int, optional): number of imgs to be passed together
                to the prediction function. Defaults to 10.
            progress_bar (bool, optional): Whether to display a progress bar or
                not. Defaults to True.
            save_imgs_path (string, optional): path of a folder where
                perturbed instances should be stored. No imgs will be stored
                if None. Defaults to None.
            load_imgs_path (string, optional): path of the folder from where
                to load saved perturbed instances. Defaults to None.
            dont_predict (bool, optional): whether to call classifier_fn or
                not. Useful if only perturbations are wanted, or want to be
                stored. Defaults to False.

        Returns:
            A numpy array of shape: [num_samples, num_classes], conaining the
            prediction of classifier_fn for each class and perturbed sample.
        """

        # Create data according to perturbation type
        data = self.get_data(num_samples)

        # Perturb imgs accordingly and run inference on them
        return data, self.perturb_and_predict(
            data, batch_size, progress_bar, save_imgs_path,
            load_imgs_path, dont_predict)

    def get_data(self, num_samples=5):
        """ Occludes multiple regions for each perturbed img (approximately
            half of the regions).

        Args:
            num_samples (int, optional): number of desired iterations of the
                perturbation loop. The number of perturbed imgs obtained is
                equal to num_features*num_samples.

        Returns:
            2d numpy array: array of shape: [num_samples+2, num_features]. This
            array will have 0 where a region is occluded for a specific
            instance, and 1 where not. The first row corresponds to the
            original img without perturbations and the second one to the
            fully occluded img.
        """

        # Initialize data
        data = np.zeros(shape=(num_samples*self.n_features, self.n_features),
                        dtype=int)

        for i in range(num_samples*self.n_features):

            # Get current feature
            i_feature = i % self.n_features

            # At the start of iteration, stablish a random order to add
            # the regions
            if i_feature == 0:
                rand_order = list(range(self.n_features))
                self.random_state.shuffle(rand_order)

            # Add the region to all next perturbed imgs of the iteration
            data[i:i+self.n_features-i_feature, rand_order[i_feature]] = 1

        # Add original img and fully occluded img at the beginning
        data = np.vstack([
            np.ones(self.n_features, dtype=int),
            np.zeros(self.n_features, dtype=int),
            data])
        return data

    def get_non_perturbed_samples(self, num_samples):
        """ Computes which of the rows in the returned data correspond to
            non-perturbed samples.

        Args:
            num_samples (int): number of desired samples when calling to
                get_data.

        Returns:
            list: list with indexs of the non perturbed samples.
        """
        return [0, 1] + [1+i*self.n_features for i in range(1, num_samples+1)]


class ConvolutionalPerturber(Perturber):

    def __init__(self, img, classifier_fn, hide_color=None,
                 random_state=None, kernel_size=[5, 5],
                 stride=[2, 2]):
        """ Perturber which occludes only one region for each perturbed img.
            The number of samples will be equal to the number of regions in the
            segmentation.

            Useful for computing the difference between occluding or not
            occluding each of the regions (ablation).

            The original img without occlusions is appended to the start.

        Args:
            img (3d numpy array): img to perturb, of shape:  [height, width, 3].
            segments (2d numpy array): segmentation of the img, of shape:
                [height, width], where each element indicates the
                region it belongs to.
            classifier_fn (_type_): function to predict the class of a img
            hide_color (int, optional): color to use occlude a region. If None,
                the mean color of a region thrughout the img is used for that
                region. Defaults to None.
            random_state (int, optional): seed to use for replicating
                perturbations. Defaults to None.
            kernel_size (2d numpy array, optional): size of the kernel to use
                in each dimension. Each value represents the fraction of the
                size in each dimension, e.g., a value of 5 would make the
                kernel be a fifth of the total size in that dimension.
            stride (2d numpy array, optional): stride to use for the kernel in each
                dimension, as the fraction of the size of the kernel in that
                dimension. 0 represents a stride of one pixel (or frame)
                and 1 represents the size of the kernel.
        """
        super().__init__(img, None, classifier_fn, hide_color,
                         random_state),
        self.kernel_size = kernel_size
        self.stride = stride

    def perturb(self, batch_size=10, progress_bar=True, save_imgs_path=None,
                load_imgs_path=None, dont_predict=False):
        """ Generates imgs with a single perturbation and their predictions.

        Args:
            batch_size (int, optional): number of imgs to be passed together
                to the prediction function. Defaults to 10.
            progress_bar (bool, optional): Whether to display a progress bar or
                not. Defaults to True.
            save_imgs_path (string, optional): path of a folder where
                perturbed instances should be stored. No imgs will be stored
                if None. Defaults to None.
            load_imgs_path (string, optional): path of the folder from where
                to load saved perturbed instances. Defaults to None.
            dont_predict (bool, optional): whether to call classifier_fn or
                not. Useful if only perturbations are wanted, or want to be
                stored. Defaults to False.

        Returns:
            A numpy array of shape: [num_samples, num_classes], conaining the
            prediction of classifier_fn for each class and perturbed sample.
        """
        # Create data according to perturbation type
        return None, self.perturb_and_predict(
            batch_size, progress_bar, save_imgs_path,
            load_imgs_path, dont_predict)

    def perturb_and_predict(self, batch_size=10, progress_bar=True,
                            save_imgs_path=None, load_imgs_path=None,
                            dont_predict=False):
        """ Generates perturbed imgs and their predictions. It has been
            adapted from the public LIME repository.

        Args:
            data (2d numpy array): array of shape: [num_samples, num_features]
                with 0 where a region is occluded for a specific instance, and
                1 where not.
            batch_size (int, optional): number of imgs to be passed together
                to the prediction function. Defaults to 10.
            progress_bar (bool, optional): Whether to display a progress bar or
                not. Defaults to True.
            save_imgs_path (string, optional): path of a folder where
                perturbed instances should be stored. No imgs will be stored
                if None. Defaults to None.
            load_imgs_path (string, optional): path of the folder from where
                to load saved perturbed instances. Defaults to None.
            dont_predict (bool, optional): whether to call classifier_fn or
                not. Useful if only perturbations are wanted, or want to be
                stored. Defaults to False.

        Returns:
            A numpy array of shape: [num_samples, num_classes], conaining the
            prediction of classifier_fn for each class and perturbed sample.
        """

        kernel_size = self.kernel_size
        stride = self.stride
        labels = []
        imgs = []

        # Progress bar
        if progress_bar:
            progress = \
                tqdm(range(int(np.prod([i*j-j+1 for i, j in
                                        zip(kernel_size, stride)])+1)))

        # Unmodified img
        imgs.append(self.img)

        for y in range(int(kernel_size[0]*stride[0]-stride[0]+1)):
            for x in range(int(kernel_size[1]*stride[1]-stride[1]+1)):

                # Generate perturbed img
                if load_imgs_path is None:

                    # Initialize perturbed img as a simple copy
                    temp = copy.deepcopy(self.img)

                    s1 = [round(a*b/(c*d)) for a, b, c, d in
                            zip([y, x], self.img.shape,
                                stride, kernel_size)]
                    s2 = [round(a+(b/c)) for a, b, c in
                            zip(s1, self.img.shape, kernel_size)]
                    temp[s1[0]:s2[0], s1[1]:s2[1], :] = \
                        self.fudged_img[s1[0]:s2[0], s1[1]:s2[1], :]

                    # Save perturbed img if specified
                    if save_imgs_path is not None:
                        save_img(temp, save_imgs_path,
                                    str(progress.n)+'.png')

                # Load perturbed img from memory
                else:
                    temp = load_img(os.path.join(load_imgs_path,
                                                    str(progress.n)+'.png'))

                # Add to the list of imgs to predict if not dont_predict
                if not dont_predict:
                    imgs.append(temp)

                # Predict all imgs in list when the batch_size is full.
                # Append results to list
                if len(imgs) == batch_size and not dont_predict:
                    preds = self.classifier_fn(np.array(imgs))
                    labels.extend(preds)
                    imgs = []
                    progress.update(batch_size)
                elif dont_predict:
                    progress.update(1)

        # Predict remaining imgs in list. Append results to list
        if len(imgs) > 0 and not dont_predict:
            preds = self.classifier_fn(np.array(imgs))
            labels.extend(preds)

            # Update progress
            progress.update(len(imgs))
            
        return np.array(labels)

    def get_data(self):
        pass

    def get_non_perturbed_samples(self, num_samples):
        """ Computes which of the rows in the returned data correspond to
            non-perturbed samples.

        Args:
            num_samples (int): number of desired samples when calling to
                get_data.

        Returns:
            list: list with indexs of the non perturbed samples.
        """
        return [0]


class ShapPerturber(Perturber):

    def __init__(self, img, segments, classifier_fn, hide_color=None,
                 random_state=None):
        """ Perturber which occludes multiple regions for each perturbed img
            (approximately half of the regions). The number of perturbed imgs
            can be chosen with num_samples.

            The original img without occlusions is appended to the start.

        Args:
            img (3d numpy array): img to perturb, of shape:  [height, width, 3].
            segments (2d numpy array): segmentation of the img, of shape:
                [height, width], where each element indicates the
                region it belongs to.
            classifier_fn (_type_): function to predict the class of a img
            hide_color (int, optional): color to use occlude a region. If None,
                the mean color of a region thrughout the img is used for that
                region. Defaults to None.
            random_state (int, optional): seed to use for replicating
                perturbations. Defaults to None.
        """
        super().__init__(img, segments, classifier_fn, hide_color,
                         random_state)

    def perturb(self, num_samples=500, batch_size=10,
                progress_bar=True, save_imgs_path=None,
                load_imgs_path=None, dont_predict=False,
                algorithm='kernel'):
        """ Generates imgs with multiple perturbations and their predictions.

        Args:
            num_samples (int, optional): number of desired perturbed imgs.
                Defaults to 500.
            exact (bool, optional): whether to perturb exactly (True) or
                approximately (False) a (1-p) proportion of the regions.
                Defaults to False.
            p (float, optional): probability for each region to be occluded.
                Defaults to 0.5.
            batch_size (int, optional): number of imgs to be passed together
                to the prediction function. Defaults to 10.
            progress_bar (bool, optional): Whether to display a progress bar or
                not. Defaults to True.
            save_imgs_path (string, optional): path of a folder where
                perturbed instances should be stored. No imgs will be stored
                if None. Defaults to None.
            load_imgs_path (string, optional): path of the folder from where
                to load saved perturbed instances. Defaults to None.
            dont_predict (bool, optional): whether to call classifier_fn or
                not. Useful if only perturbations are wanted, or want to be
                stored. Defaults to False.

        Returns:
            A numpy array of shape: [num_samples, num_classes], conaining the
            prediction of classifier_fn for each class and perturbed sample.
        """
        if algorithm == 'kernel':
            explainer = shap.KernelExplainer(
                partial(self.perturb_and_predict,
                        batch_size=batch_size,
                        progress_bar=progress_bar,
                        save_imgs_path=save_imgs_path,
                        load_imgs_path=load_imgs_path,
                        dont_predict=dont_predict),
                np.zeros((1, self.n_features)))
            self.shapley_values = explainer.shap_values(
                np.ones((1, self.n_features)), nsamples=num_samples)
        elif algorithm == 'permutation':
            explainer = shap.Explainer(
                partial(self.perturb_and_predict,
                        batch_size=batch_size,
                        progress_bar=progress_bar,
                        save_imgs_path=save_imgs_path,
                        load_imgs_path=load_imgs_path,
                        dont_predict=dont_predict),
                np.zeros((1, self.n_features)),
                algorithm='permutation')
            self.shapley_values = explainer(
                np.ones((1, self.n_features)), max_evals=num_samples)

        return self.data, self.labels

    def explain(self, label_to_explain=None,  algorithm='kernel'):

        # Label to explain
        if label_to_explain is None:
            label_to_explain = np.argmax(self.classifier_fn(
                np.array([self.img]))[0])

        if algorithm == 'kernel':
            return list(zip(range(self.n_features),
                            self.shapley_values[label_to_explain][0]))
        if algorithm == 'partition':
            return list(zip(range(self.n_features),
                            self.shapley_values.values[0, :, label_to_explain]))
        # return self.shapley_values