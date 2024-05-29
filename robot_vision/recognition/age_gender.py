# INSIGHTFACE
from insightface.app import FaceAnalysis

# MiVOLO
from mivolo.predictor import Predictor

from robot_vision.recognition.recognizer import Recognizer
from robot_vision.utils.plotting import draw_detections

import numpy as np
import copy
import random

class AgeGender(Recognizer):

    GENDERS = ['female', 'male']

    def __init__(self) -> None:
        pass

    def get_age_gender(self, img):
        """Find age and gender of person in image. Depending on the method, a face detector is needed.

        Args:
            img (np.ndarray): image where to detect the face keypoints, in BGR.

        Returns:
            (age: int, gender: str): age and gender of the detected person/face.
            If multiple persons/faces are found, only the age and gender of one of them are returned.
        """
        pass

    def get_result(self, img):
        return self.get_age_gender(img)

    @staticmethod
    def get_plot_result(img, result):
        age, gender = result
        return draw_detections(img, age=age, gender=gender)


class InsightFaceAgeGender(AgeGender):
    """ Model: buffalo_l: SCRFD-10GF for detection and ResNet50@WebFace600K for recognition.

    Github: https://github.com/deepinsight/insightface
    """
        
    def __init__(self, det_thresh=.5, det_size=(64, 64)) -> None:
        super().__init__()
            
        self.app = FaceAnalysis(providers=['CUDAExecutionProvider', 'CPUExecutionProvider'], allowed_modules = ['detection', 'genderage',])
        self.app.prepare(ctx_id=0, det_thresh=det_thresh, det_size=det_size)

    def get_age_gender(self, img: np.ndarray):

        faces = self.app.get(img)

        # No faces
        if len(faces) < 1:
            return None

        return round(faces[0]['age']), AgeGender.GENDERS[faces[0]['gender']]
    
    def get_explanation(self, img, explainer_fn, label=None, mode='gender', face_bbox=None, num_samples=1000):
        """Get explanation for the prediction.

        Args:
            img (3d numpy array): image to explain.
            explainer_fn (function): explainer function to call, receiving the image, the prediction function, the label and the number of samples.
            label (int, optional): class to explain. If None, class with highest prediction confidence explained. Defaults to None.
            mode (str, optional): Whether to explain age or gender. Currently, only 'gender' supported. Defaults to 'gender'.
            face_bbox (1d numpy array, optional): If provided, perturbations only made in the delimited region. Defaults to None.
            num_samples (int, optional): Number of perturbed samples to use for explanation. Defaults to 1000.

        Raises:
            ValueError: If mode != 'gender'.

        Returns:
            3d numpy array: RGB explanation image
        """

        if face_bbox is None:
            face_img = img
        else:
            face_img = img[face_bbox[1]:face_bbox[3], face_bbox[0]:face_bbox[2]]
        
        # Get explanation
        if mode == 'gender':
            self.bbox = face_bbox
            self.img = img
            return explainer_fn(face_img, self.pred_fn, label, num_samples=num_samples)
        else:
            raise ValueError('Mode not supported.')
        
    def pred_fn(self, imgs):
        
        res = []
        for i in range(imgs.shape[0]):

            if self.bbox is None:
                img = imgs[i,...]
            else:
                img = copy.deepcopy(self.img)
                img[self.bbox[1]:self.bbox[3], self.bbox[0]:self.bbox[2]] = imgs[i,...]

            faces = self.app.get(img)

            if len(faces) < 1:
                pred = random.randint(0, len(AgeGender.GENDERS) - 1)
                # print('No face found, replaced with random prediction.')
            else:
                pred = faces[0]['gender']
            
            # Find index of element in list
            aux = [0, 0]
            aux[pred] = 1
            res.append(aux)
            
        return np.array(res)



class MiVOLOAgeGender(AgeGender):
    """MiVOLO: YOLOv8 + Multi-Input ViT.

    Github: https://github.com/WildChlamydia/MiVOLO
    """
        
    def __init__(self, detector_weights_path, mivolo_checkpoint_path, device='cuda:0', with_persons=True, disable_faces=False, verbose=False) -> None:
        super().__init__()

        config = MiVOLOAgeGender.Config(
            detector_weights=detector_weights_path,
            checkpoint=mivolo_checkpoint_path,
            device=device,
            with_persons=with_persons,
            disable_faces=disable_faces,
            draw = False
        )

        self.predictor = Predictor(config, verbose=verbose)

    def get_age_gender(self, img: np.ndarray):

        detected_objects, _ = self.predictor.recognize(img)

        # No faces
        if detected_objects.n_faces + detected_objects.n_persons < 1:
            return None

        return round(detected_objects.ages[0]), detected_objects.genders[0]
    
    def get_explanation(self, img, explainer_fn, label=None, mode='gender', face_bbox=None):
        """Get explanation for the prediction.

        Args:
            img (3d numpy array): image to explain.
            explainer_fn (function): explainer function to call, receiving the image, the prediction function, the label and the number of samples.
            label (int, optional): class to explain. If None, class with highest prediction confidence explained. Defaults to None.
            mode (str, optional): Whether to explain age or gender. Currently, only 'gender' supported. Defaults to 'gender'.
            face_bbox (1d numpy array, optional): If provided, perturbations only made in the delimited region. Defaults to None.
            num_samples (int, optional): Number of perturbed samples to use for explanation. Defaults to 1000.

        Raises:
            ValueError: If mode != 'gender'.

        Returns:
            3d numpy array: RGB explanation image
        """

        if face_bbox is None:
            face_img = img
        else:
            face_img = img[face_bbox[1]:face_bbox[3], face_bbox[0]:face_bbox[2]]
        
        # Get explanation
        if mode == 'gender':
            self.bbox = face_bbox
            self.img = img
            self.predictions = self.predictor.detector.predict(img)
            return explainer_fn(face_img, self.pred_fn, label)
        else:
            raise ValueError('Mode not supported.')
        
    def pred_fn(self, imgs):
        
        res = []
        for i in range(imgs.shape[0]):

            if self.bbox is None:
                img = imgs[i,...]
            else:
                img = copy.deepcopy(self.img)
                img[self.bbox[1]:self.bbox[3], self.bbox[0]:self.bbox[2]] = imgs[i,...]

            self.predictor.age_gender_model.predict(img, self.predictions)
            
            # Find index of element in list
            aux = [0, 0]
            aux[AgeGender.GENDERS.index(self.predictions.genders[0])] = 1
            res.append(aux)
            
        return np.array(res)

    class Config():
        def __init__(self, detector_weights, checkpoint, device, with_persons, disable_faces, draw):
            self.detector_weights = detector_weights
            self.checkpoint = checkpoint
            self.device = device
            self.with_persons = with_persons
            self.disable_faces = disable_faces
            self.draw = draw

