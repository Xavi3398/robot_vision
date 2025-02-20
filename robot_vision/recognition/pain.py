import numpy as np
import cv2

from robot_vision.utils.preprocessing import BBoxPreprocessor, KeypointsPreprocessor
from robot_vision.utils.plotting import draw_detections_faces
from robot_vision.utils import nets
from robot_vision.recognition.keypoints import Keypoints
from robot_vision.recognition.detection import Detector
from robot_vision.recognition.recognizer import Recognizer

import matplotlib.pyplot as plt

class PainRecognizer(Recognizer):
    
    PAIN_EXPRESSIONS = ['no_pain', 'pain']

    def __init__(self) -> None:
        pass

    def get_pain(self, img) -> str:
        pass

    def get_result(self, img):
        return self.get_pain(img)

    @staticmethod
    def get_plot_result(img, faces):
        return draw_detections_faces(img, faces)


class KerasPainRecognizer(PainRecognizer):

    MODEL_NAMES = ['SilNet', 'WeiNet', 'AlexNet', 'SongNet', 'InceptionV3',
                    'VGG19', 'VGG16', 'ResNet50', 'ResNet101V2', 'Xception',
                    'MobileNetV3Large', 'EfficientNetV2B0']

    def __init__(self, model_name, weights, face_detector: Detector=None, keypoints_detector: Keypoints=None, 
                 preprocessing_grayscale=False, preprocessing_mode='reflect') -> None:
        super().__init__()

        if model_name not in KerasPainRecognizer.MODEL_NAMES:
            raise KeyError(model_name + ' is not a valid model. Available models: ' + str(KerasPainRecognizer.MODEL_NAMES))

        self.model, self.img_size = nets.getNetByNamePain(model_name, n_classes=2)
        self.model.load_weights(weights)

        if face_detector is not None:
            self.detector = face_detector
            self.detector_key = 'detection'
            self.preprocessor = BBoxPreprocessor(img_size=self.img_size, grayscale=preprocessing_grayscale, mode=preprocessing_mode)
        elif keypoints_detector is not None:
            self.detector = keypoints_detector
            self.detector_key = 'keypoints'
            self.preprocessor = KeypointsPreprocessor(img_size=self.img_size, grayscale=preprocessing_grayscale, mode=preprocessing_mode)
        else:
            self.preprocessor = None

    def get_pain(self, img) -> str:
        
        # Preprocess image
        if self.preprocessor is not None:
            faces = self.detector.get_result(img)
        else:
            faces = [None]
        
        # Case where no face found
        if len(faces) < 1:
            return []

        results = []

        for face in faces:

            # Preprocess image
            if self.preprocessor is not None:
                img = self.preprocessor.preprocess(img, face[self.detector_key])
        
            # Case where no face found
            if img is None:
                continue

            # Convert to RGB if necessary
            if len(img.shape) < 3 or img.shape[2] < 3:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

            # Get prediction
            predictions = self.model.predict(np.array([img]))[0]

            # Get expression
            results.append({**face, 'pain': PainRecognizer.PAIN_EXPRESSIONS[np.argmax(predictions)]})

        return results
    
    def get_explanation(self, img, explainer_fn, label=None):
        
        # Preprocess image
        if self.preprocessor is not None:
            faces = self.detector.get_result(img)
        else:
            faces = [None]
        
        # Case where no face found
        if len(faces) < 1:
            return []

        results = []

        for face in faces:

            # Preprocess image
            if self.preprocessor is not None:
                img = self.preprocessor.preprocess(img, face[self.detector_key])
        
            # Case where no face found
            if img is None:
                continue

            # Convert to RGB if necessary
            if len(img.shape) < 3 or img.shape[2] < 3:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            
            # Get explanation
            results.append(explainer_fn(img, self.model, label))
        
        return results