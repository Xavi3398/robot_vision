import cv2
import numpy as np
import math
from scipy import ndimage
import matplotlib.pyplot as plt

class Preprocessing:

    EYE_L = 0
    EYE_R = 1
    NOSE = 2
    MOUTH_L = 3
    MOUTH_R = 4

    def __init__(self, keypoints_detector, img_size=150, mode='reflect'):
        self.img_size = img_size
        self.mode = mode
        self.keypoints_detector = keypoints_detector

    def preprocess(self, img, debug=False):

        keypoints = self.keypoints_detector.get_keypoints(img)

        # Case where no face found
        if keypoints is None:
            return None

        # Too many keypoints. We do not know which ones are from the eyes, nose and mouth
        if len(keypoints) > 5:
            raise Exception('Keypoints detector must return either the eye landmarks (mode "eyes") or the landmarks of the eyes, mouth corners and nose (mode "5").')

        # Only eyes keypoints. Estimate nose and mouth.
        # Mouth coordinates: x=eyes, y=eyes+eye_distance
        # Nose coordinates: mean(eye_l, eye_r, mouth_l, mouth_r)
        if len(keypoints) == 2:
            eyes_keypoints = keypoints
            eye_distance = magnitude(get_vector(eyes_keypoints[0], eyes_keypoints[1]))
            mouth_keypoints = [[eyes_keypoints[1][0], eyes_keypoints[0][1] + eye_distance], [eyes_keypoints[1][0], eyes_keypoints[0][1] + eye_distance]]
            nose_keypoint = [eyes_keypoints[0][0] + eye_distance/2, np.mean(eyes_keypoints[0][1], eyes_keypoints[1][1]) + eye_distance/2]
            keypoints = [eyes_keypoints[0], eyes_keypoints[1], nose_keypoint, mouth_keypoints[0], mouth_keypoints[1]] 

        # To grayscale
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 

        # Shift
        middle_point = (img.shape[1]//2, img.shape[0]//2)
        img_gray = ndimage.shift(img_gray, (middle_point[1] - keypoints[Preprocessing.NOSE][1], middle_point[0] - keypoints[Preprocessing.NOSE][0]), mode=self.mode)
        keypoints = translate_points(keypoints, (middle_point[0] - keypoints[Preprocessing.NOSE][0], middle_point[1] - keypoints[Preprocessing.NOSE][1]))

        # Rotate
        angle = get_eye_rotation(keypoints[Preprocessing.EYE_L], keypoints[Preprocessing.EYE_R])
        img_gray = ndimage.rotate(img_gray, radians_to_degrees(angle), reshape=False, mode=self.mode)
        keypoints = rotate_points(keypoints, angle, middle_point)

        # Crop
        padding_h = (keypoints[Preprocessing.MOUTH_L][1] - keypoints[Preprocessing.EYE_L][1]) / 2
        height = width = int(4 * padding_h)
        eye_d = keypoints[Preprocessing.EYE_R][0] - keypoints[Preprocessing.EYE_L][0]
        padding_w = (width - eye_d) / 2
        p1_x = int(keypoints[Preprocessing.EYE_L][0] - padding_w)
        p1_y = int(keypoints[Preprocessing.EYE_L][1] - padding_h)
        p2_x = p1_x + width
        p2_y = p1_y + height
        bbox = (p1_x, p1_y, p2_x, p2_y)
        img_gray = crop_img(img_gray, bbox)
        keypoints = translate_points(keypoints, (-p1_x, -p1_y))

        # Resize
        zoom = self.img_size / img_gray.shape[0]
        img_gray = ndimage.zoom(img_gray, zoom, mode=self.mode)
        keypoints = resize_points(keypoints, zoom)

        # Normalize
        img_gray = img_to_float(img_gray)
        img_gray = histogram_stretching(img_gray)
        img_gray = img_to_uint8(img_gray)

        if debug:
            img_gray = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)
            for kp in keypoints:
                cv2.circle(img_gray, (int(kp[0]), int(kp[1])), radius=int(self.img_size/30), color=(0,0,255), thickness=-1)
                
        return img_gray

def get_eye_rotation(eye1, eye2):
    v_eyes = get_vector(eye1, eye2)
    angle = get_angle(v_eyes, (1, 0))
    return angle if v_eyes[1] > 0 else -angle

def get_angle(v1, v2):
    """Get angle between two vectors."""
    return math.acos(np.dot(v1, v2) / (magnitude(v1) * magnitude(v2)))

def get_unit_vector(v):
    """Normalize vector"""
    m = magnitude(v)
    return (v[0]/m, v[1]/m)
    
def get_vector(p1, p2):
    """Get vector between two points."""
    return get_unit_vector((p2[0] - p1[0], p2[1] - p1[1]))

def magnitude(v): 
    """Get magnitude of a vector."""
    return math.sqrt(sum(pow(x, 2) for x in v))

def crop_img(img, bbox):
    """Crop an image by a bbox."""
    return img[bbox[1]:bbox[3], bbox[0]:bbox[2], ...]

def resize_points(keypoints, zoom):
    return [(kp[0] * zoom, kp[1] * zoom) for kp in keypoints]

def translate_points(keypoints, movement):
    return [(kp[0] + movement[0], kp[1] + movement[1]) for kp in keypoints]
    
def rotate_points(keypoints, radians, center):
    """Rotate all keypoints around the origin (0, 0)."""
    
    # Translate to origin
    keypoints = translate_points(keypoints, (-center[0], -center[1]))
    
    # Rotate each point around origin
    keypoints = [rotate_point_origin(kp, radians) for kp in keypoints]
    
    # Translate back
    keypoints = translate_points(keypoints, (center[0], center[1]))
    return keypoints
    
def rotate_point_origin(xy, radians):
    """Rotate a point around the origin (0, 0)."""
    x, y = xy
    xx = x * math.cos(radians) + y * math.sin(radians)
    yy = -x * math.sin(radians) + y * math.cos(radians)

    return xx, yy

def radians_to_degrees(rad):
    return rad * 180 / math.pi

def histogram_stretching(img, h_min=0, h_max=1):
    max_value = np.max(img)
    min_value = np.min(img)
    if max_value > 0 and min_value != max_value:
        return h_min+(h_max-h_min)*(img-min_value)/(max_value-min_value)
    else:
        return img
    
def img_to_uint8(img):
    return (img * 255).astype('uint8')

def img_to_float(img):
    return img / 255

def bbox_xy2wh(bbox_xy):
    return np.array([bbox_xy[0], bbox_xy[1], bbox_xy[2] - bbox_xy[0], bbox_xy[3] - bbox_xy[1]])

def bbox_wh2xy(bbox_xy):
    return np.array([bbox_xy[0], bbox_xy[1], bbox_xy[0] + bbox_xy[2], bbox_xy[1] + bbox_xy[3]])