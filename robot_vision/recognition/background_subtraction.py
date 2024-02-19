# RemBG
from rembg import new_session, remove

import numpy as np
import cv2

from robot_vision.recognition.recognizer import Recognizer
from robot_vision.utils.plotting import draw_detections

class BackgroundSubtractor(Recognizer):

    def __init__(self) -> None:
        pass

    def remove_background(self, img) -> np.ndarray:
        """Find face bounding box in image.

        Args:
            img (_type_): image where to detect the face, in BGR.

        Returns:
            np.ndarray: bounding box of the detected face in the image in the format [x0, y0, x1, y1], 
            where (x0, y0) is the upper left point and (x1, y1) is the lower right point of the box. 
            If various faces are found, only one is returned.
        """
        pass

    def get_result(self, img):
        return self.remove_background(img)

    @staticmethod
    def get_plot_result(img, result):
        background_subtraction = result
        return draw_detections(background_subtraction)

class RemBGBackgroundSubtractor(BackgroundSubtractor):

    def __init__(self, session_name='u2net_human_seg', bgcolor=None, only_mask=False) -> None:
        super().__init__()
        self.session = new_session(session_name)
        self.bgcolor = bgcolor
        self.only_mask = only_mask

    def remove_background(self, img) -> np.ndarray:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = remove(img, session=self.session, bgcolor=self.bgcolor, only_mask=self.only_mask)

        if not self.only_mask:
            img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGRA)
        return img