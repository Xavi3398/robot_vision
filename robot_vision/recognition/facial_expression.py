import numpy as np
import cv2

from robot_vision.utils.preprocessing import bbox_wh2xy, Preprocessing
from robot_vision.utils.plotting import draw_detections
from robot_vision.utils import nets
from robot_vision.recognition.keypoints import Keypoints
from robot_vision.recognition.recognizer import Recognizer

import matplotlib.pyplot as plt

class FacialExpressionRecognizer(Recognizer):
    
    FACIAL_EXPRESSIONS = ['anger', 'disgust', 'fear', 'happiness', 'sadness', 'surprise']

    def __init__(self) -> None:
        pass

    def get_facial_expression(self, img) -> np.ndarray:
        """Find bounding box of target class in image.

        Args:
            img (_type_): image where to perform detection, in BGR.

        Returns:
            np.ndarray: bounding box of the detection in the format [x0, y0, x1, y1], 
            where (x0, y0) is the upper left point and (x1, y1) is the lower right point of the box. 
            If various detections are found, only one is returned.
        """
        pass

    def get_result(self, img):
        return self.get_facial_expression(img)

    @staticmethod
    def get_plot_result(img, result):
        expression = result
        return draw_detections(img, expression=expression)


class KerasFacialExpressionRecognizer(FacialExpressionRecognizer):
    """
    """

    MODEL_NAMES = ['SilNet', 'WeiNet', 'AlexNet', 'SongNet', 'InceptionV3',
                    'VGG19', 'VGG16', 'ResNet50', 'ResNet101V2', 'Xception',
                    'MobileNetV3Large', 'EfficientNetV2B0']

    def __init__(self, model_name, weights, keypoints_detector: Keypoints) -> None:
        super().__init__()

        if model_name not in KerasFacialExpressionRecognizer.MODEL_NAMES:
            raise KeyError(model_name + ' is not a valid model. Available models: ' + str(KerasFacialExpressionRecognizer.MODEL_NAMES))

        self.model, self.img_size = nets.getNetByName(model_name)
        self.model.load_weights(weights)

        if keypoints_detector is not None:
            self.preprocessor = Preprocessing(keypoints_detector=keypoints_detector, img_size=self.img_size)
        else:
            self.preprocessor = None

    def get_facial_expression(self, img) -> np.ndarray:
        
        if self.preprocessor is not None:
            img = self.preprocessor.preprocess(img)
        
        # Case where no face found
        if img is None:
            return None

        if len(img.shape) < 3 or img.shape[2] < 3:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

        predictions = self.model.predict(np.array([img]))[0]

        return FacialExpressionRecognizer.FACIAL_EXPRESSIONS[np.argmax(predictions)]