# SPIGA
from spiga.inference.config import ModelConfig
from spiga.inference.framework import SPIGAFramework

# INSIGHTFACE
from insightface.app import FaceAnalysis

# FaceNet's MTCNN
import mtcnn

# DLIB
import dlib
from imutils import face_utils

import numpy as np
import cv2

from robot_vision.recognition.detection import Detector
from robot_vision.recognition.recognizer import Recognizer
from robot_vision.utils.preprocessing import bbox_xy2wh, crop_img
from robot_vision.utils.plotting import draw_detections


class Keypoints(Recognizer):

    MODES = ['eyes', '5', 'all']

    def __init__(self, mode='5') -> None:
        self.change_mode(mode)

    def change_mode(self, mode):
        
        if mode not in Keypoints.MODES:
            raise KeyError(mode + ' is not a valid mode. Available modes: ' + str(Keypoints.MODES))
        
        self.mode = mode

    def get_keypoints(self, img) -> np.ndarray:
        """Find keypoints of face in image. Depending on the method, a face detector is needed.

        Args:
            img (np.ndarray): image where to detect the face keypoints, in BGR.

        Returns:
            np.ndarray: keypoints bounding box of the detected face in the image in the format [EYE_L, EYE_R, NOSE, MOUTH_L, MOUTH_R], 
            where each element is a tuple with the point coordinates as(x, y). 
            If various faces are found, only the keypoints of one of them are returned.
        """
        pass

    def five2eyes(keypoints_5):
        return np.array([keypoints_5[0], keypoints_5[1]])

    def get_result(self, img):
        return self.get_keypoints(img)

    @staticmethod
    def get_plot_result(img, result):
        keypoints = result
        return draw_detections(img, kps=keypoints)


class SPIGAKeypoints(Keypoints):
    """ SPIGA: Shape Preserving Facial Landmarks with Graph Attention Networks.
    
    Github: https://github.com/andresprados/SPIGA
    """
        
    def __init__(self, face_detector: Detector, mode: str = 'all') -> None:
        super().__init__(mode)

        self.face_detector = face_detector
        dataset = 'wflw'
        self.processor = SPIGAFramework(ModelConfig(dataset))

    def get_keypoints(self, img: np.ndarray):

        # Face bbox needed
        face_bbox = self.face_detector.get_bbox(img)
        face_bbox = bbox_xy2wh(face_bbox)

        features = self.processor.inference(img, [face_bbox])
        landmarks = np.array(features['landmarks'][0])
        
        if self.mode == '5' or self.mode == 'eyes':
            keypoints_5 = [np.mean(landmarks[60:68], axis=0), np.mean(landmarks[68:76], axis=0), landmarks[53], landmarks[88], landmarks[92]]
            if self.mode == '5':
                return keypoints_5
            else:
                return Keypoints.five2eyes(keypoints_5)
        elif self.mode == 'all':
            return landmarks
        else:
            raise KeyError(self.mode + ' is not a valid mode. Available modes: ' + str(Keypoints.MODES))


class InsightFaceKeypoints(Keypoints):
    """ Only face detection. Model: buffalo_l: SCRFD-10GF for detection and ResNet50@WebFace600K for recognition.

    Github: https://github.com/deepinsight/insightface
    """
        
    def __init__(self, mode: str = 'all', det_thresh=.5, det_size=(64, 64)) -> None:
        super().__init__(mode)

        if mode == '5' or mode == 'eyes':
            allowed_modules = ['detection']
        else:
            allowed_modules = ['detection', 'landmark_2d_106']

        self.app = FaceAnalysis(providers=['CUDAExecutionProvider', 'CPUExecutionProvider'], allowed_modules=allowed_modules)
        self.app.prepare(ctx_id=0, det_thresh=det_thresh, det_size=det_size)

    def get_keypoints(self, img: np.ndarray):

        faces = self.app.get(img)

        # No faces
        if len(faces) < 1:
            return None
        
        if self.mode == '5' or self.mode == 'eyes':
            keypoints_5 = faces[0]['kps']
            if self.mode == '5':
                return keypoints_5
            else:
                return Keypoints.five2eyes(keypoints_5)
        elif self.mode == 'all':
            return faces[0]['landmark_2d_106']
        else:
            raise KeyError(self.mode + ' is not a valid mode. Available modes: ' + str(Keypoints.MODES))


class MTCNNKeypoints(Keypoints):
    """Multi-Task CNN.

    Github: https://github.com/ipazc/mtcnn
    """

    def __init__(self, weights_file, mode: str = '5', min_face_size: int = 20, steps_threshold: list = None, scale_factor: float = 0.709) -> None:
        """
        Initializes the MTCNN.
        :param weights_file: file uri with the weights of the P, R and O networks from MTCNN. By default it will load
        the ones bundled with the package.
        :param min_face_size: minimum size of the face to detect
        :param steps_threshold: step's thresholds values
        :param scale_factor: scale factor
        """
        super().__init__(mode=mode)

        self.face_detector = mtcnn.MTCNN(weights_file, min_face_size, steps_threshold, scale_factor)

    def get_keypoints(self, img: np.ndarray) -> np.ndarray:

        face_detector = mtcnn.MTCNN()
        faces = face_detector.detect_faces(img)

        # No faces
        if len(faces) < 1:
            return None

        if self.mode == '5' or self.mode == 'all' or self.mode == 'eyes':
            keypoints = faces[0]['keypoints']
            keypoints_5 = np.array([keypoints['left_eye'], keypoints['right_eye'], keypoints['nose'], 
                keypoints['mouth_left'],keypoints['mouth_right']])
                            
            if self.mode == 'eyes':
                return Keypoints.five2eyes(keypoints_5)
            else:
                return keypoints_5
        else:
            raise KeyError(self.mode + ' is not a valid mode. Available modes: ' + str(Keypoints.MODES))


class ViolaJonesKeypoints(Keypoints):
    """Viola Jones eye detection. Optionally run first face detection to reduce false positives.
    
    OpenCV: https://docs.opencv.org/3.4/db/d28/tutorial_cascade_classifier.html
    """

    def __init__(self, xml_path, face_detector: Detector = None, scaleFactor=1.1, minNeighors=3, minSize=None, maxSize=None) -> None:
        super().__init__(mode='eyes')
        
        self.face_cascade = cv2.CascadeClassifier(xml_path)
        self.face_detector = face_detector
        self.scaleFactor = scaleFactor
        self.minNeighors = minNeighors
        self.minSize = minSize
        self.maxSize = maxSize

    
    def get_keypoints(self, img: np.ndarray) -> np.ndarray:

        if self.face_detector is not None:
            bbox = self.face_detector.get_bbox()
            img_crop = crop_img(img, bbox)
        else:
            img_crop = img

        gray = cv2.cvtColor(img_crop, cv2.COLOR_BGR2GRAY)
        eyes = self.face_cascade.detectMultiScale(gray, self.scaleFactor, self.minNeighors, self.minSize, self.maxSize)

        if len(eyes) < 2:
            print('Less than two eyes detected')
            return None
        
        if len(eyes) > 2:
            print('More than two eyes detected')
            return None
        
        eye1 = ViolaJonesKeypoints.get_eye_center(eyes[0])
        eye2 = ViolaJonesKeypoints.get_eye_center(eyes[1])

        if self.face_detector is not None:
            eye1[0] += bbox[0]
            eye1[1] += bbox[1]
            eye2[0] += bbox[0]
            eye2[1] += bbox[1]

        if eye1[0] <= eye2[0]:
            return np.array([eye1, eye2])
        else:
            return np.array([eye2, eye1])

    def get_eye_center(bbox):
        x, y, w, h = bbox
        return [x+w/2, y+h/2]


class DLIBKeypoints(Keypoints):
    """ DLIB facial landmark detector: implementation of "One Millisecond Face Alignment with an Ensemble of Regression Trees".

    Python API: http://dlib.net/python/index.html.
    
    Tutorial: https://pyimagesearch.com/2017/04/03/facial-landmarks-dlib-opencv-python/
    """

    def __init__(self, predictor_path, face_detector: Detector, mode: str = 'all') -> None:
        super().__init__(mode)
        
        self.face_detector = face_detector
        self.predictor = dlib.shape_predictor(predictor_path)
    
    def get_keypoints(self, img: np.ndarray) -> np.ndarray:

        bbox = self.face_detector.get_bbox(img)

        # Detecta los landmarks de la imagen
        dlib_rect = dlib.rectangle(bbox[0], bbox[1], bbox[2], bbox[3])
        keypoints = self.predictor(img, dlib_rect)
        keypoints = face_utils.shape_to_np(keypoints)

        # TODO
        if self.mode == '5' or self.mode == 'eyes':
            keypoints_5 = [np.mean(keypoints[36:42], axis=0), np.mean(keypoints[42:48], axis=0), keypoints[30], keypoints[48], keypoints[54]]
            if self.mode == '5':
                return keypoints_5
            else:
                return Keypoints.five2eyes(keypoints_5)
        elif self.mode == 'all':
            return keypoints
        else:
            raise KeyError(self.mode + ' is not a valid mode. Available modes: ' + str(Keypoints.MODES))