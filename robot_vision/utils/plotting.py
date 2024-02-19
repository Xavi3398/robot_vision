import cv2
import numpy as np
import copy
import matplotlib.pyplot as plt
from scipy import ndimage


def show_img(img: np.ndarray):
    if len(img.shape) > 1:
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    else:
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_GRAY2RGB))

def draw_bbox(img: np.ndarray, bbox, color=[255,0,0], thickness=2):
    bbox = bbox.astype('int')
    img = cv2.line(img, [bbox[0], bbox[1]], [bbox[2], bbox[1]], color, thickness) 
    img = cv2.line(img, [bbox[0], bbox[1]], [bbox[0], bbox[3]], color, thickness) 
    img = cv2.line(img, [bbox[2], bbox[3]], [bbox[2], bbox[1]], color, thickness) 
    img = cv2.line(img, [bbox[2], bbox[3]], [bbox[0], bbox[3]], color, thickness)
    return img

def draw_keypoints(img: np.ndarray, keypoints, color=[255,0,0], radius=3, thickness=-1):
    for kp in keypoints:
        img = cv2.circle(img, (int(kp[0]), int(kp[1])), radius=radius, color=color, thickness=thickness)
    return img

def draw_text(img: np.ndarray, point, text, color=[255,0,0], thickness=1, font=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1):
    return cv2.putText(img, text, point, font, fontScale, color, thickness, cv2.LINE_AA)

def draw_detections(img: np.ndarray, bbox=None, kps=None, age=None, gender=None, expression=None, user_face=None):

    # Recognition
    if user_face is not None:
        
        img2 = cv2.imread(user_face)

        # Zoom 2nd image to fit image dimensions
        zoom = min([img.shape[0] / img2.shape[0], img.shape[1] / img2.shape[1]])
        img2 = ndimage.zoom(img2, [zoom, zoom, 1])

        # Add black padding
        aux = np.zeros(shape=img.shape, dtype='uint8')
        aux[:min([img2.shape[0], img.shape[0]]),:min([img2.shape[1], img.shape[1]]),:] = img2[:min([img2.shape[0], img.shape[0]]),:min([img2.shape[1], img.shape[1]]),:]
        img = aux

        # # Add padding to smaller image
        # if img2.shape[0] < img.shape[0]:
        #     aux = np.zeros(shape=[img.shape[0], img2.shape[1], img.shape[2]], dtype='uint8')
        #     aux[:img2.shape[0], :img2.shape[1], :img2.shape[2]] = img2
        #     img2 = aux
        # elif img2.shape[0] > img.shape[0]:
        #     aux = np.zeros(shape=[img2.shape[0], img.shape[1], img2.shape[2]], dtype='uint8')
        #     aux[:img.shape[0], :img.shape[1], :img.shape[2]] = img
        #     img = aux

        # img = cv2.hconcat([img, img2])
    
    # BGR to RGB
    img = copy.deepcopy(img[:,:,::-1])
    
    # Bounding box
    if bbox is not None:
        img = draw_bbox(img, bbox)
        
    # Keypoints
    if kps is not None:
        img = draw_keypoints(img, kps)
    
    # Age & Gender
    if age is not None or gender is not None or expression is not None:
        
        font_factor = img.shape[1]/400
        point_factor = img.shape[1]//300

        elements = []
        for element in [age, gender, expression]:
            if element is not None:
                elements.append(element)
        
        text = str(elements.pop(0))
        while len(elements) > 0:
            text += ', ' + str(elements.pop(0))
            
        if bbox is None:
            point = [10, 10]
        else:
            point = bbox[:2].astype('int')
        img = draw_text(img, [point[0]+5*point_factor, point[1]+25*point_factor], text, fontScale=font_factor)
    
    # RGB to BGR
    img = img[:,:,::-1]

    return img