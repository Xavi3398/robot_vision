# INSIGHTFACE
from insightface.app import FaceAnalysis

import numpy as np
import cv2
from scipy.spatial import distance

from robot_vision.recognition.recognizer import Recognizer
from robot_vision.utils.plotting import draw_detections

class FacialRecognition(Recognizer):

    def __init__(self, distance_fn=distance.cosine) -> None:
        self.distance_fn = distance_fn
    
    def initialize_identities(self, img_paths) -> None:
        """Initializes the identities, where the recognized images will be searched in.

        Args:
            img_paths (list of strings): list with the paths of the images to use as identities.
        """

        self.rec_ids = []
        
        for img_path in img_paths:
            img = cv2.imread(img_path)
            features = self.get_features(img)

            if features is None:
                print('Error initializing faces.')
                raise Exception()

            self.rec_ids.append({'img_path': img_path, 'features': features})
        
        print(str(len(self.rec_ids)) + ' identities initialized.')

    def recognize(self, img) -> int:
        """Extracts visual features from the image and compares them to that of the initialized identities.

        Args:
            img (np.ndarray): image where to detect the features, in BGR.

        Returns:
            id (int): index of the recocognized identity.
        """
        features = self.get_features(img)

        # Case where no face found
        if features is None:
            return None

        distances = [self.distance_fn(features, id['features']) for id in self.rec_ids]
        return self.rec_ids[np.argmin(distances)]['img_path']

    def get_features(self, img) -> np.ndarray:
        """Find features in image. Depending on the method, a face detector is needed.

        Args:
            img (np.ndarray): image where to detect the features, in BGR.

        Returns:
            features (np.ndarray): vector of visual features.
        """
        pass

    def get_result(self, img):
        return self.recognize(img)

    @staticmethod
    def get_plot_result(img, result):
        user_face = result
        return draw_detections(img, user_face=user_face)


class InsightFaceRecognition(FacialRecognition):
    """ Model: buffalo_l: SCRFD-10GF for detection and ResNet50@WebFace600K for recognition.

    Github: https://github.com/deepinsight/insightface
    """
        
    def __init__(self, img_paths, distance_fn=distance.cosine, det_thresh=.5, det_size=(64, 64)) -> None:
        super().__init__(distance_fn)
            
        self.app = FaceAnalysis(providers=['CUDAExecutionProvider', 'CPUExecutionProvider'], allowed_modules = ['detection', 'recognition'])
        self.app.prepare(ctx_id=0, det_thresh=det_thresh, det_size=det_size)

        self.initialize_identities(img_paths)

    def get_features(self, img: np.ndarray) -> np.ndarray:

        faces = self.app.get(img)

        # No faces
        if len(faces) < 1:
            return None

        return faces[0]['embedding']
