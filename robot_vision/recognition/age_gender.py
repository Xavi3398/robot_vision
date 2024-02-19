# INSIGHTFACE
from insightface.app import FaceAnalysis

# MiVOLO
from mivolo.predictor import Predictor

from robot_vision.recognition.recognizer import Recognizer
from robot_vision.utils.plotting import draw_detections

import numpy as np

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

    class Config():
        def __init__(self, detector_weights, checkpoint, device, with_persons, disable_faces, draw):
            self.detector_weights = detector_weights
            self.checkpoint = checkpoint
            self.device = device
            self.with_persons = with_persons
            self.disable_faces = disable_faces
            self.draw = draw

