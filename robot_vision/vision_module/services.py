
import base64
import numpy as np
import cv2

from multiprocessing import Process, Pipe, Queue, Lock
from multiprocessing.connection import Connection

from robot_vision.recognition.recognizer import Recognizer
from robot_vision.recognition import predefined

predefined.MODELS_FOLDER = '../models'
predefined.USER_FACES_FOLDER = '../user_faces'


def process_images(conn: Connection, input_frames: Queue, output_results: Queue):

    # To start, wait until we receive a task and method. MAY BLOCK UNTIL WE RECEIVE ELEMENTS
    task, method, plot = conn.recv()
    recognizer = predefined.predefined_selector(task, method)

    while True:
        
        # If we receive new task/method: change recognizer
        if conn.poll():
            new_task, new_method, plot = conn.recv()

            if new_task != task or new_method != method:
                print('------ AI THREAD: Changing recognizer -------')
                recognizer = predefined.predefined_selector(new_task, new_method)
            
            task = new_task
            method = new_method

        # Get frame from the input queue. MAY BLOCK UNTIL THERE ARE ELEMENTS
        img = input_frames.get()
            
        print('------ AI THREAD: New computation -------')

        result = recognizer.get_result(img)

        # Return only recognition
        if plot:
            plot_img = recognizer.get_plot_result(img, result)
        else:
            plot_img = None
        
        # Remove first element if queue is full. No wait so as to not block in case that between full() and get() someone removed all elements
        if output_results.full():
            try:
                output_results.get_nowait()
            except:
                pass
        
        # Put the result in the output queue
        output_results.put((result, plot_img))


def image_to_base64(img):

    # Encode the processed image as a JPEG-encoded base64 string
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 90]
    _, frame_encoded = cv2.imencode(".jpg", img, encode_param)
    processed_img_data = base64.b64encode(frame_encoded).decode()

    # Prepend the base64-encoded string with the data URL prefix
    b64_src = "data:image/jpg;base64,"
    processed_img_data = b64_src + processed_img_data
    return processed_img_data


def base64_to_image(base64_string):

    # Extract the base64 encoded binary data from the input string
    base64_data = base64_string.split(",")[1]

    # Decode the base64 data to bytes
    image_bytes = base64.b64decode(base64_data)

    # Convert the bytes to numpy array
    image_array = np.frombuffer(image_bytes, dtype=np.uint8)

    # Decode the numpy array as an image using OpenCV
    image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
    return image


# def process_image(img, method, task, plot=True):
    
#     # Initizlize recognizer if task or method changed
#     if CurrentRecognizer.task != task or CurrentRecognizer.method != method:
#         recognizer: Recognizer = predefined_selector(task, method)
#         CurrentRecognizer.update(task, method, recognizer)
    
#     # Process img with AI
#     recognizer = CurrentRecognizer.recognizer
#     result = recognizer.get_result(img)

#     # Return only recognition
#     if not plot:
#         return result, None
    
#     # Return resulting plot
#     plot_img = recognizer.get_plot_result(img, result)
#     return result, plot_img