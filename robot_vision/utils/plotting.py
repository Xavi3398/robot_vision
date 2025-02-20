import cv2
import numpy as np
import copy
import matplotlib.pyplot as plt
from scipy import ndimage


def show_img(img: np.ndarray):
    plt.axis('off')
    if len(img.shape) > 1:
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    else:
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_GRAY2RGB))
    plt.show()

def draw_bbox(img: np.ndarray, bbox, color=[255,0,0], thickness=2):
    bbox = bbox.astype('int')
    img = cv2.line(img, [bbox[0], bbox[1]], [bbox[2], bbox[1]], color, thickness) 
    img = cv2.line(img, [bbox[0], bbox[1]], [bbox[0], bbox[3]], color, thickness) 
    img = cv2.line(img, [bbox[2], bbox[3]], [bbox[2], bbox[1]], color, thickness) 
    img = cv2.line(img, [bbox[2], bbox[3]], [bbox[0], bbox[3]], color, thickness)
    return img

def draw_keypoints(img: np.ndarray, keypoints, color=[255,0,0], radius=3, thickness=-1, numbers=False):
    for i, kp in enumerate(keypoints):
        if numbers:
            img = cv2.putText(img, str(i), (int(kp[0]), int(kp[1])), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color=color, thickness=1)
        else:
            img = cv2.circle(img, (int(kp[0]), int(kp[1])), radius=radius, color=color, thickness=thickness)
    return img

def draw_text(img: np.ndarray, point, text, color=(255, 0, 0), color_bg=(255, 255, 255), thickness=2, font=cv2.FONT_HERSHEY_PLAIN, fontScale=2):
    x, y = point

    text_size, _ = cv2.getTextSize(text, font, fontScale, thickness)
    text_w, text_h = text_size
    padding = .5 * text_h

    cv2.rectangle(img, (point[0], int(point[1] + padding/2)), (int(x + text_w), int(y - text_h - padding/2)), color_bg, -1)
    cv2.putText(img, text, point, font, fontScale, color, thickness)

    return img

def draw_detections_faces(img: np.ndarray, faces):


    for face in faces:

        bbox = face['detection'] if 'detection' in face else None
        kps = face['keypoints'] if 'keypoints' in face else None
        age = face['age'] if 'age' in face else None
        gender = face['gender'] if 'gender' in face else None
        expression = face['expression'] if 'expression' in face else None
        user_face = face['face_recognition'] if 'face_recognition' in face else None
        mouth_open = face['mouth_open'] if 'mouth_open' in face else None
        pain = face['pain'] if 'pain' in face else None

        img = draw_detections(img, bbox=bbox, kps=kps, age=age, gender=gender, expression=expression, user_face=user_face, mouth_open=mouth_open, pain=pain)

    return img

def draw_detections(img: np.ndarray, bbox=None, kps=None, age=None, gender=None, expression=None, user_face=None, mouth_open=None, pain=None):

    font_factor = img.shape[1]/300
    point_factor = img.shape[1]//300

    # Recognition
    if user_face is not None:
        img2 = cv2.imread(user_face)

        bbox2 = bbox.copy().astype('int')
        bbox2[0] = max(bbox2[0], 0)
        bbox2[1] = max(bbox2[1], 0)
        bbox2[2] = min(bbox2[2], img.shape[1])
        bbox2[3] = min(bbox2[3], img.shape[0])

        # Zoom 2nd image to fit image dimensions
        zoom_x = (bbox2[2] - bbox2[0])/img2.shape[1]
        zoom_y = (bbox2[3] - bbox2[1])/img2.shape[0]
        img2 = ndimage.zoom(img2, [zoom_y, zoom_x, 1])
        
        # Paste 2nd image in bbox of 1st image
        img[bbox2[1]:bbox2[3], bbox2[0]:bbox2[2]] = img2
    
    # BGR to RGB
    img = copy.deepcopy(img[:,:,::-1])
    
    # Bounding box
    if bbox is not None:
        img = draw_bbox(img, bbox)
        
    # Keypoints
    if kps is not None:
        img = draw_keypoints(img, kps)
    
    # Age & Gender
    if age is not None or gender is not None or expression is not None or pain is not None:

        elements = []
        for element in [age, gender, expression, pain]:
            if element is not None:
                elements.append(element)
        
        text = str(elements.pop(0))
        while len(elements) > 0:
            text += ', ' + str(elements.pop(0))
            
        if bbox is None:
            point = [10, 10]
        else:
            point = bbox[:2].astype('int')
        img = draw_text(img, [point[0]+5*point_factor, point[1]+15*point_factor], text, fontScale=font_factor)
    
    # Mouth open
    if mouth_open is not None:

        # Draw mouth points
        img = draw_keypoints(img, [mouth_open[1], mouth_open[2]])

        # Draw line between mouth points
        img = cv2.line(img, (int(mouth_open[1][0]), int(mouth_open[1][1])), (int(mouth_open[2][0]), int(mouth_open[2][1])), (255, 0, 0), 2)

        # Draw text
        point = mouth_open[2].astype(int)
        point[0] += 10
        img = draw_text(img, point, str(round(mouth_open[0], 2)), fontScale=font_factor)

    # RGB to BGR
    img = img[:,:,::-1]

    return img