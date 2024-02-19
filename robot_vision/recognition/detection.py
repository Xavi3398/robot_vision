# YOLO
from ultralytics.yolo.engine.model import YOLO

# INSIGHTFACE
from insightface.app import FaceAnalysis

# FaceNet's MTCNN
import mtcnn

# DLIB
import dlib

import numpy as np
import cv2

from robot_vision.recognition.recognizer import Recognizer
from robot_vision.utils.preprocessing import bbox_wh2xy
from robot_vision.utils.plotting import draw_detections

class Detector(Recognizer):

    def __init__(self) -> None:
        pass

    def get_bbox(self, img) -> np.ndarray:
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
        return self.get_bbox(img)

    @staticmethod
    def get_plot_result(img, result):
        bbox = result
        return draw_detections(img, bbox=bbox)


class YOLODetector(Detector):
    """ YOLOv8. Performs detection for Person (0) and Face (1) classes.

    WebPage: https://docs.ultralytics.com/
    """


    def __init__(self, weights, det_class, conf=.4, iou=.7, half=True, verbose=False) -> None:
        super().__init__()

        self.det_class = det_class
        self.yolo = YOLO(weights)
        self.yolo.fuse()
        self.conf = conf
        self.iou = iou
        self.half = half
        self.verbose = verbose

    def get_bbox(self, img) -> np.ndarray:
        
        yolo_pred = self.yolo(img, conf=self.conf, iou=self.iou, half=self.half, verbose=self.verbose)
        
        if len(yolo_pred) < 1:
            return None
        
        yolo_pred = yolo_pred[0].boxes
        classes = yolo_pred.cls.numpy(force=True)
        bboxes = yolo_pred.xyxy.numpy(force=True)
        
        if not self.det_class in classes:
            return None
        
        return bboxes[np.where(classes == self.det_class)[0]][0].astype('int')


class InsightFaceFaceDetector(Detector):
    """ Only face detection. Model: buffalo_l: SCRFD-10GF for detection and ResNet50@WebFace600K for recognition.

    Github: https://github.com/deepinsight/insightface
    """
        
    def __init__(self, det_thresh=.5, det_size=(64, 64)) -> None:
        super().__init__()

        self.app = FaceAnalysis(providers=['CUDAExecutionProvider', 'CPUExecutionProvider'], allowed_modules=['detection'])
        self.app.prepare(ctx_id=0, det_thresh=det_thresh, det_size=det_size)

    def get_bbox(self, img: np.ndarray) -> np.ndarray:

        faces = self.app.get(img)

        # No faces
        if len(faces) < 1:
            return None
            
        return faces[0]['bbox']


class MTCNNFaceDetector(Detector):
    """Multi-Task CNN. Only face detection.

    Github: https://github.com/ipazc/mtcnn
    """

    def __init__(self, weights_file, min_face_size: int = 20, steps_threshold: list = None, scale_factor: float = 0.709) -> None:
        """
        Initializes the MTCNN.
        :param weights_file: file uri with the weights of the P, R and O networks from MTCNN. By default it will load
        the ones bundled with the package.
        :param min_face_size: minimum size of the face to detect
        :param steps_threshold: step's thresholds values
        :param scale_factor: scale factor
        """
        super().__init__()

        self.face_detector = mtcnn.MTCNN(weights_file, min_face_size, steps_threshold, scale_factor)
    
    def get_bbox(self, img: np.ndarray) -> np.ndarray:

        faces = self.face_detector.detect_faces(img)

        # No faces
        if len(faces) < 1:
            return None

        face_wh = faces[0]['box']
        return bbox_wh2xy(face_wh)


class ViolaJonesFaceDetector(Detector):
    """Viola Jones face detection.
    
    OpenCV: https://docs.opencv.org/3.4/db/d28/tutorial_cascade_classifier.html
    """

    def __init__(self, xml_path, scaleFactor=1.1, minNeighors=3, minSize=None, maxSize=None) -> None:
        super().__init__()
        
        self.face_cascade = cv2.CascadeClassifier(xml_path)
        self.scaleFactor = scaleFactor
        self.minNeighors = minNeighors
        self.minSize = minSize
        self.maxSize = maxSize
    
    def get_bbox(self, img: np.ndarray) -> np.ndarray:

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, self.scaleFactor, self.minNeighors, self.minSize, self.maxSize)

        # No faces
        if len(faces) < 1:
            return None

        face_wh = faces[0]
        return bbox_wh2xy(face_wh)


class DLIBFaceDetector(Detector):
    """ DLIB face detector: HOG + Linear SVM.

    Python API: http://dlib.net/python/index.html.

    Tutorial: https://pyimagesearch.com/2017/04/03/facial-landmarks-dlib-opencv-python/
    """

    def __init__(self) -> None:
        super().__init__()

        self.detector = dlib.get_frontal_face_detector()

    
    def get_bbox(self, img: np.ndarray) -> np.ndarray:

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = self.detector(gray, 1)

        # No faces
        if len(faces) < 1:
            return None

        face_wh = faces[0]
        return np.array([face_wh.left(), face_wh.top(), face_wh.right(), face_wh.bottom()])