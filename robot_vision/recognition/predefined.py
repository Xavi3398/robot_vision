from robot_vision.recognition import detection
from robot_vision.recognition import facial_recognition
from robot_vision.recognition import background_subtraction
from robot_vision.recognition import age_gender
from robot_vision.recognition import keypoints
from robot_vision.recognition import facial_expression
from robot_vision.recognition import mouth_open

import os

MODELS_FOLDER = 'robot_vision/models'
USER_FACES_FOLDER = 'robot_vision/user_faces'


def predefined_selector(task, method):
    
    # Inexistent task
    if task not in PREDEFINED_RECOGNIZERS.keys():
        raise Exception('Invalid task ('+task+'). Must be one of:'+str(PREDEFINED_RECOGNIZERS.keys()))

    # Inexistent method
    if method not in PREDEFINED_RECOGNIZERS[task].keys():
        raise Exception('Invalid method ('+method+') for task: ('+task+'). Must be one of:'+str(PREDEFINED_RECOGNIZERS[task].keys()))
    
    # Inexistent task
    if task not in PREDEFINED_RECOGNIZERS.keys():
        raise Exception('Invalid task ('+task+'). Must be one of:'+str(PREDEFINED_RECOGNIZERS.keys()))
    
    return PREDEFINED_RECOGNIZERS[task][method]()


# FACE DETECTION
def predefined_face_detection_YOLOv8():
    return detection.YOLODetector(
            weights=os.path.join(MODELS_FOLDER, 'yolov8x_person_face.pt'), det_class=1)

def predefined_face_detection_InsightFace():
    return detection.InsightFaceFaceDetector()

def predefined_face_detection_MTCNN():
    return detection.MTCNNFaceDetector(
            weights_file=os.path.join(MODELS_FOLDER, 'mtcnn_weights.npy'))

def predefined_face_detection_ViolaJones():
    return detection.ViolaJonesFaceDetector(
            xml_path=os.path.join(MODELS_FOLDER, 'haarcascade_frontalface_alt.xml'))

def predefined_face_detection_DLIB():
    return detection.DLIBFaceDetector()
    

# PERSON DETECTION
def predefined_person_detection_YOLOv8():
    return detection.YOLODetector(
            weights=os.path.join(MODELS_FOLDER, 'yolov8x_person_face.pt'), det_class=0)


# FACE KEYPOINTS
def predefined_keypoints_SPIGA():
    return keypoints.SPIGAKeypoints(
            face_detector=PREDEFINED_RECOGNIZERS['face_detection'][list(PREDEFINED_RECOGNIZERS['face_detection'].keys())[0]]())

def predefined_keypoints_InsightFace():
    return keypoints.InsightFaceKeypoints()

def predefined_keypoints_MTCNN():
    return keypoints.MTCNNKeypoints(
            weights_file=os.path.join(MODELS_FOLDER, 'mtcnn_weights.npy'))

def predefined_keypoints_ViolaJones():
    return keypoints.ViolaJonesKeypoints(
            xml_path=os.path.join(MODELS_FOLDER, 'haarcascade_eye.xml'))

def predefined_keypoints_DLIB():
    return keypoints.DLIBKeypoints(
            predictor_path=os.path.join(MODELS_FOLDER, 'shape_predictor_68_face_landmarks.dat'),
            face_detector=PREDEFINED_RECOGNIZERS['face_detection'][list(PREDEFINED_RECOGNIZERS['face_detection'].keys())[0]]())

# FACIAL EXPRESSION RECOGNITION
def predefined_facial_expression_recognition_Keras(model_name):
    return facial_expression.KerasFacialExpressionRecognizer(
            model_name=model_name,
            weights=os.path.join(MODELS_FOLDER, 'expression_recognition', model_name+'_CV1_weights.h5'),
            keypoints_detector=keypoints.InsightFaceKeypoints(mode='5'))

def predefined_facial_expression_recognition_AlexNet():
    return predefined_facial_expression_recognition_Keras('AlexNet')

def predefined_facial_expression_recognition_SilNet():
    return predefined_facial_expression_recognition_Keras('SilNet')

def predefined_facial_expression_recognition_SongNet():
    return predefined_facial_expression_recognition_Keras('SongNet')

def predefined_facial_expression_recognition_WeiNet():
    return predefined_facial_expression_recognition_Keras('WeiNet')

def predefined_facial_expression_recognition_ResNet50():
    return predefined_facial_expression_recognition_Keras('ResNet50')

def predefined_facial_expression_recognition_ResNet101V2():
    return predefined_facial_expression_recognition_Keras('ResNet101V2')

def predefined_facial_expression_recognition_VGG16():
    return predefined_facial_expression_recognition_Keras('VGG16')

def predefined_facial_expression_recognition_VGG19():
    return predefined_facial_expression_recognition_Keras('VGG19')

def predefined_facial_expression_recognition_Xception():
    return predefined_facial_expression_recognition_Keras('Xception')

def predefined_facial_expression_recognition_InceptionV3():
    return predefined_facial_expression_recognition_Keras('InceptionV3')

def predefined_facial_expression_recognition_MobileNetV3Large():
    return predefined_facial_expression_recognition_Keras('MobileNetV3Large')

def predefined_facial_expression_recognition_EfficientNetV2B0():
    return predefined_facial_expression_recognition_Keras('EfficientNetV2B0')


# AGE & GENDER ESTIMATION
def predefined_age_gender_MiVOLO():
    return age_gender.MiVOLOAgeGender(
            detector_weights_path=os.path.join(MODELS_FOLDER, 'yolov8x_person_face.pt'), 
            mivolo_checkpoint_path=os.path.join(MODELS_FOLDER, 'mivolo_imbd.pth.tar'))

def predefined_age_gender_InsightFace():
    return age_gender.InsightFaceAgeGender()


# FACE RECOGNITION
def predefined_face_recognition_InsightFace():
    return facial_recognition.InsightFaceRecognition([os.path.join(USER_FACES_FOLDER, img_name) for img_name in os.listdir(USER_FACES_FOLDER)])


# BACKGROUND SUBTRACTION
def predefined_background_subtraction_RemBG():
    return background_subtraction.RemBGBackgroundSubtractor()
    
# MOUTH OPEN
def predefined_mouth_open_InsightFace(**args):
    return mouth_open.InsightFaceMouthOpen(**args)

def predefined_mouth_open_SPIGA():
    return mouth_open.SPIGAMouthOpen(
            face_detector=PREDEFINED_RECOGNIZERS['face_detection'][list(PREDEFINED_RECOGNIZERS['face_detection'].keys())[0]]())


PREDEFINED_RECOGNIZERS = {
    'face_detection': {
        'YOLOv8': predefined_face_detection_YOLOv8,
        'InsightFace': predefined_face_detection_InsightFace,
        'MTCNN': predefined_face_detection_MTCNN,
        'ViolaJones': predefined_face_detection_ViolaJones,
        'DLIB': predefined_face_detection_DLIB},
    'person_detection': {
        'YOLOv8': predefined_person_detection_YOLOv8},
    'keypoints':  {
        'SPIGA': predefined_keypoints_SPIGA,
        'InsightFace': predefined_keypoints_InsightFace,
        'MTCNN': predefined_keypoints_MTCNN,
        'ViolaJones': predefined_keypoints_ViolaJones,
        'DLIB': predefined_keypoints_DLIB},
    'expression': {
        'AlexNet': predefined_facial_expression_recognition_AlexNet,
        'SilNet': predefined_facial_expression_recognition_SilNet,
        'SongNet': predefined_facial_expression_recognition_SongNet,
        'WeiNet': predefined_facial_expression_recognition_WeiNet,
        'ResNet50': predefined_facial_expression_recognition_ResNet50,
        'ResNet101V2': predefined_facial_expression_recognition_ResNet101V2,
        'VGG16': predefined_facial_expression_recognition_VGG16,
        'VGG19': predefined_facial_expression_recognition_VGG19,
        'Xception': predefined_facial_expression_recognition_Xception,
        'InceptionV3': predefined_facial_expression_recognition_InceptionV3,
        'MobileNetV3Large': predefined_facial_expression_recognition_MobileNetV3Large,
        'EfficientNetV2B0': predefined_facial_expression_recognition_EfficientNetV2B0},
    'age_gender': {
        'MiVOLO': predefined_age_gender_MiVOLO,
        'InsightFace': predefined_age_gender_InsightFace},
    'face_recognition': {
        'InsightFace': predefined_face_recognition_InsightFace},
    'background_subtraction': {
        'RemBG': predefined_background_subtraction_RemBG},
    'mouth_open': {
        'InsightFace': predefined_mouth_open_InsightFace,
        'SPIGA': predefined_mouth_open_SPIGA}
}