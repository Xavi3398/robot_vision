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
from skimage import transform

from robot_vision.recognition.detection import Detector
from robot_vision.recognition.recognizer import Recognizer
from robot_vision.utils.preprocessing import bbox_xy2wh, crop_img
from robot_vision.utils.plotting import draw_detections


class MouthOpen(Recognizer):

    MODES = ['2d', '3d']

    def __init__(self, mode='2d', factor=4) -> None:
        self.change_mode(mode)
        self.factor = factor

    def change_mode(self, mode):
        
        if mode not in MouthOpen.MODES:
            raise KeyError(mode + ' is not a valid mode. Available modes: ' + str(MouthOpen.MODES))
        
        self.mode = mode

    def get_mouth_open(self, img) -> np.ndarray:
        """Find distance between upper and lower lip.

        Args:
            img (np.ndarray): image where to detect the open mouth, in BGR.

        Returns:
            tuple (distance, upper_point, lower_point): distance of the mouth and coordinates of the upper and lower lip, 
            where each point is a tuple with the coordinates (x, y). 
            If various faces are found, only the keypoints of one of them are returned.
        """
        pass

    def get_mouth_open_aux(self, keypoints, eye_l, eye_r, mouth_t, mouth_b, ear_l, ear_r):
        """Find distance between upper and lower lip.

        Args:
            keypoints (np.ndarray): keypoints of the face.
            eye_l (int): index of the left eye.
            eye_r (int): index of the right eye.
            mouth_l (int): index of the left mouth.
            mouth_r (int): index of the right mouth.

        Returns:
            tuple (distance, upper_point, lower_point): distance of the mouth and coordinates of the upper and lower lip, 
            where each point is a tuple with the coordinates (x, y). 
        """

        mouth = [keypoints[mouth_t], keypoints[mouth_b]]
        eyes = [keypoints[eye_l] if type(eye_l) == int else np.mean(keypoints[eye_l], axis=0), keypoints[eye_r] if type(eye_r) == int else np.mean(keypoints[eye_r], axis=0)]
        
        # Homography
        if self.homography_dst_points is not None:
            ears = [keypoints[ear_l], keypoints[ear_r]]
            homography_src_points = [eyes[0], eyes[1], ears[0], ears[1]]
            tform = transform.estimate_transform('projective', np.array(homography_src_points, dtype='float32'), np.array(self.homography_dst_points, dtype='float32'))
            tf_mouth = tform(np.array(mouth))
            tf_eyes = tform(np.array(eyes))
            mouth_distance = self.factor * np.linalg.norm(tf_mouth[1] - tf_mouth[0]) / np.linalg.norm(tf_mouth[0] - np.mean(tf_eyes, axis=0))

        # No homography
        else:
            mouth_distance = self.factor * np.linalg.norm(mouth[1] - mouth[0]) / np.linalg.norm(mouth[0] - np.mean(eyes, axis=0))
        
        return [mouth_distance, mouth[0][:2], mouth[1][:2]]

    def get_result(self, img):
        return self.get_mouth_open(img)

    @staticmethod
    def get_plot_result(img, result):
        return draw_detections(img, mouth_open=result)


class InsightFaceMouthOpen(MouthOpen):
    """ Only face detection. Model: buffalo_l: SCRFD-10GF for detection, ResNet50@WebFace600K for recognition, MobileNet for age_gender and facial keypoints.

    Github: https://github.com/deepinsight/insightface
    """
        
    def __init__(self, mode: str = '2d', det_thresh=.5, det_size=(64, 64), factor=4, use_homography=False) -> None:
        super().__init__(mode, factor)

        if mode == '2d':
            allowed_modules = ['detection', 'landmark_2d_106']
        elif mode == '3d':
            allowed_modules = ['detection', 'landmark_3d_68']
        else:
            raise KeyError(mode + ' is not a valid mode. Available modes: ' + str(MouthOpen.MODES))

        self.app = FaceAnalysis(providers=['CUDAExecutionProvider', 'CPUExecutionProvider'], allowed_modules=allowed_modules)
        self.app.prepare(ctx_id=0, det_thresh=det_thresh, det_size=det_size)

        # 2D keypoints
        if self.mode == '2d':
            self.mouth_t = 71
            self.mouth_b = 53
            self.eye_l = [33, 34, 35, 36, 37, 38, 39, 40, 41, 42]
            self.eye_r = [87, 88, 89, 90, 91, 92, 93, 94, 95, 96]
            # self.ear_l = 52 # Mouth_left
            # self.ear_r = 61 # Mouth_right
            self.ear_l = 16
            self.ear_r = 32
        
        # 3D keypoints
        elif self.mode == '3d':
            self.mouth_t = 51
            self.mouth_b = 57
            self.eye_l = [36, 37, 38, 39, 40, 41]
            self.eye_r = [42, 43, 44, 45, 46, 47]
            self.ear_l = 3
            self.ear_r = 13
        
        else:
            raise KeyError(mode + ' is not a valid mode. Available modes: ' + str(MouthOpen.MODES))

        if use_homography:

            # Read normal image
            img_path = '../../notebooks/resources/mouth_open_front.jpg'
            # img_path = 'resources/mouth_open_front.jpg'
            img_normal = cv2.imread(img_path)

            # Detect faces
            faces = self.app.get(img_normal)

            # No faces
            if len(faces) < 1:
                print('No faces detected')
                return None

            # Get keypoints
            if self.mode == '2d':
                keypoints_normal = faces[0]['landmark_2d_106']
            elif self.mode == '3d':
                keypoints_normal = faces[0]['landmark_3d_68']
            else:
                raise KeyError(mode + ' is not a valid mode. Available modes: ' + str(MouthOpen.MODES))

            # Homography keypoints
            self.homography_dst_points = [np.mean(keypoints_normal[self.eye_l], axis=0), np.mean(keypoints_normal[self.eye_l], axis=0), keypoints_normal[self.ear_l], keypoints_normal[self.ear_r]]
            
        else:
            self.homography_dst_points = None

    def get_mouth_open(self, img: np.ndarray):

        faces = self.app.get(img)

        # No faces
        if len(faces) < 1:
            return None
        
        # 2D keypoints
        if self.mode == '2d':
            keypoints = faces[0]['landmark_2d_106']
        
        # 3D keypoints
        elif self.mode == '3d':
            keypoints = faces[0]['landmark_3d_68']

        # Invalid mode
        else:
            raise KeyError(self.mode + ' is not a valid mode. Available modes: ' + str(MouthOpen.MODES))
        
        return self.get_mouth_open_aux(keypoints, self.eye_l, self.eye_r, self.mouth_t, self.mouth_b, self.ear_l, self.ear_r)


class SPIGAMouthOpen(MouthOpen):
    """ Only face detection. Model: buffalo_l: SCRFD-10GF for detection, ResNet50@WebFace600K for recognition, MobileNet for age_gender and facial keypoints.

    Github: https://github.com/deepinsight/insightface
    """
        
    def __init__(self, face_detector: Detector, factor=4, use_homography=False) -> None:
        super().__init__('2d', factor)
        
        self.face_detector = face_detector
        dataset = 'wflw'
        self.processor = SPIGAFramework(ModelConfig(dataset))
        
        self.mouth_t = 79
        self.mouth_b = 85
        self.eye_l = [60, 61, 62, 63, 64, 65, 66, 67]
        self.eye_r = [68, 69, 70, 71, 72, 73, 74, 75]
        # self.ear_l = 5
        # self.ear_r = 27
        self.ear_l = 76 # Mouth_left
        self.ear_r = 82 # Mouth_right

        if use_homography:

            # Read normal image
            img_path = '../../notebooks/resources/mouth_open_front.jpg'
            # img_path = 'resources/mouth_open_front.jpg'
            img_normal = cv2.imread(img_path)

            # Detect faces
            face_bbox = self.face_detector.get_bbox(img_normal)

            # No faces
            if face_bbox is None:
                print('No faces detected')
                return None

            # Get keypoints
            features = self.processor.inference(img_normal, [face_bbox])
            keypoints_normal = np.array(features['landmarks'][0])

            # Homography keypoints
            self.homography_dst_points = [np.mean(keypoints_normal[self.eye_l], axis=0), np.mean(keypoints_normal[self.eye_l], axis=0), keypoints_normal[self.ear_l], keypoints_normal[self.ear_r]]
            
        else:
            self.homography_dst_points = None

    def get_mouth_open(self, img: np.ndarray):

        # Detect faces
        face_bbox = self.face_detector.get_bbox(img)
        face_bbox = bbox_xy2wh(face_bbox)

        # No faces
        if face_bbox is None:
            print('No faces detected')
            return None

        # Get keypoints
        features = self.processor.inference(img, [face_bbox])
        keypoints = np.array(features['landmarks'][0])

        return self.get_mouth_open_aux(keypoints, self.eye_l, self.eye_r, self.mouth_t, self.mouth_b, self.ear_l, self.ear_r)
