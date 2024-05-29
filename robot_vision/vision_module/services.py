
import base64
import numpy as np
import cv2
import traceback 

from multiprocessing import Process, Pipe, Queue, Lock
from multiprocessing.connection import Connection

from robot_vision.recognition import predefined as predefined_recognition
from robot_vision.explanation import predefined as predefined_explanation
from robot_vision import explanation

predefined_recognition.MODELS_FOLDER = '../models'
predefined_recognition.USER_FACES_FOLDER = '../user_faces'


def process_images(conn: Connection, input_frames: Queue, output_results: Queue):
    
    print('------- AI THREAD: Start -------')

    # To start, wait until we receive a task and method. MAY BLOCK UNTIL WE RECEIVE ELEMENTS
    task, method, mode, expl_method = conn.recv()
    recognizer = predefined_recognition.predefined_selector(task, method)
    if mode == 'explanation':
        explainer = predefined_explanation.predefined_selector(expl_method)

    while True:

        # Get frame from the input queue. MAY BLOCK UNTIL THERE ARE ELEMENTS
        img = input_frames.get()
        
        # If we receive new task/method: change recognizer
        if conn.poll():
            new_task, new_method, new_mode, new_expl_method = conn.recv()

            if new_task != task or new_method != method:
                print('------- AI THREAD: Changing recognizer -------')
                recognizer = predefined_recognition.predefined_selector(new_task, new_method)
            
            if mode == 'explanation' and new_expl_method != expl_method:
                explainer = predefined_explanation.predefined_selector(new_expl_method)
            
            task = new_task
            method = new_method
            mode = new_mode
            expl_method = new_expl_method
            
        print('------- AI THREAD: New computation -------')

        if mode == 'explanation':
            try:
                result = recognizer.get_result(img)
                expl_img = recognizer.get_explanation(img, explainer)
            except Exception as error:
                print('------- AI THREAD: Error in explanation: -------')
                print(error)
                print(traceback.print_exc() )
                print('------- AI THREAD: End of error -------')

                result = None
                plot_img = None
        else:
            try:
                result = recognizer.get_result(img)
            except Exception as error:
                print('------- AI THREAD: Error in recognition: -------')
                print(error)
                print(traceback.print_exc() )
                print('------- AI THREAD: End of error -------')

                result = None

        # Return also plot or not
        if mode == 'explanation' and result is not None and expl_img is not None:
            plot_img = recognizer.get_plot_result(expl_img, result)
        elif mode == 'plot' and result is not None:
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
        output_results.put((None, plot_img))


def image_to_base64(img):

    # Encode the processed image as a JPEG-encoded base64 string
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 90]
    _, frame_encoded = cv2.imencode(".jpg", img, encode_param)
    processed_img_data = base64.b64encode(frame_encoded)
    return processed_img_data


def base64_to_image(base64_data):

    # Decode the base64 data to bytes
    image_bytes = base64.b64decode(base64_data)

    # Convert the bytes to numpy array
    image_array = np.frombuffer(image_bytes, dtype=np.uint8)

    # Decode the numpy array as an image using OpenCV
    image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
    return image
