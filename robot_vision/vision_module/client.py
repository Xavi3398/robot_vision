from time import sleep
import cv2
import socketio #python-socketio by @miguelgrinberg
import services
import threading
import requests

from robot_vision.recognition.predefined import PREDEFINED_RECOGNIZERS

base_url = 'http://127.0.0.1:8080'

sio = socketio.Client()
sio.connect(base_url, namespaces='/videoStream')


@sio.event
def connect():
    print('connection established')


@sio.event
def disconnect():
    print('disconnected from server')


def get_as_options(methods):
    text = '\t0 - default'
    for i, method in enumerate(methods):
        text += '\n\t' + str(i+1) + ' - ' + method
    return text


def ask_recognitions():

    print('Starting AI Recognition module')

    while not RecordingInfo.end:

        choice = 0
        while choice not in [1,2,3,4]:
            print('\nAvailable modes:'+
                '\n1 - (video stream): Plot recognitions in real time.'+
                '\n2 - (video stream): Return recognitions as text.'+
                '\n3 - (screen capture): Plot recognitions in real time.'+
                '\n4 - (screen capture): Return recognitions as text.')
            choice = int(input('\nChoose mode: '))

        stream_mode = 'video' if choice in [1,2] else 'image'
        mode = 'plot' if choice in [1,3] else 'text'

        print('\nAvailable tasks:')
        print(get_as_options(list(PREDEFINED_RECOGNIZERS.keys())))
        choice = int(input('\nIntroduce a task: ')) - 1
        task = list(PREDEFINED_RECOGNIZERS.keys())[choice if choice > 0 else 0]
        
        print('\nAvailable methods:')
        print(get_as_options(list(PREDEFINED_RECOGNIZERS[task].keys())))
        choice = int(input('\nIntroduce a method: ')) - 1
        method = list(PREDEFINED_RECOGNIZERS[task].keys())[choice if choice > 0 else 0]

        # Clear buffer of frames of server. No funciona bien
        # response = requests.post(base_url+'/cleanBuffer')
        # print(response.text)

        print('\nStarting recording. Press "q" or RIGHT MOUSE BUTTON to stop.\n')

        if stream_mode == 'image':
            print('Press LEFT MOUSE BUTTON for screen capture.\n')

        record_and_send_webcam(task, method, mode, stream_mode)

        continue_recognition = ''
        while continue_recognition not in ['y', 'n', 'yes', 'no']:
            continue_recognition = input('Run new recognition (y/n)?').lower()
        
        if continue_recognition not in['y', 'yes']:
            RecordingInfo.end = True

    # Close OpenCV windows
    if RecordingInfo.showingRecordingWindow:
        cv2.waitKey(1)
        cv2.destroyWindow('Recording')
    
    return



def record_and_send_webcam(task, method, mode, stream_mode):

    RecordingInfo.stop_video_feed = False
    cam = cv2.VideoCapture(0)

    while (not RecordingInfo.stop_video_feed):

        # get frame from webcam
        ret, frame = cam.read()

        # Close
        if RecordingInfo.stop_video_feed or(cv2.waitKey(1) & 0xFF == ord('q')):
            RecordingInfo.stop_video_feed = True
        
        # Show recording
        if RecordingInfo.showRecording:
            cv2.imshow('Recording', frame) 
            
            if not RecordingInfo.showingRecordingWindow:
                RecordingInfo.showingRecordingWindow = True
                cv2.setMouseCallback('Recording', mouseEvent)
        
        # send to server
        if stream_mode == 'video':
            img_64_out = services.image_to_base64(frame)
            sio.emit('sendFrame', (img_64_out, task, method, mode), namespace='/videoStream')
        elif RecordingInfo.capture:
            RecordingInfo.capture = False
            img_64_out = services.image_to_base64(frame)
            response = requests.post(base_url+'/sendImage?task='+task+'&method='+method+'&mode='+mode, data=img_64_out, headers={'content-type': 'image/jpg'})
            if mode == 'plot':
                img_response = services.base64_to_image(response.text)
                RecordingInfo.processed_frame = img_response
            elif mode == 'text':
                print(response.text)

    # Release resources
    cam.release()


@sio.on('sendPlot', namespace='/videoStream')
def receive_and_show_plot(img_64_in):
    img = services.base64_to_image(img_64_in)
    RecordingInfo.processed_frame = img


@sio.on('sendText', namespace='/videoStream')
def receive_and_show_text(message):
    print('Result received: ' + message)
    

def mouseEvent(event,x,y,flags,param):
    if event == cv2.EVENT_LBUTTONUP:
        RecordingInfo.capture = True
    if event == cv2.EVENT_RBUTTONUP:
        RecordingInfo.stop_video_feed = True


def show_processed_feed():
        
    while (not RecordingInfo.end):
        
        if RecordingInfo.stop_video_feed:
            sleep(.100)
        else:

            # Close
            if RecordingInfo.stop_video_feed or(cv2.waitKey(1) & 0xFF == ord('q')):
                RecordingInfo.stop_video_feed = True

            # Show recording
            if RecordingInfo.processed_frame is not None and not RecordingInfo.stop_video_feed:

                cv2.imshow('Processing', RecordingInfo.processed_frame)
                
                if not RecordingInfo.showingProcessingWindow:
                    RecordingInfo.showingProcessingWindow = True
                    cv2.setMouseCallback('Processing',mouseEvent)
        
    # Close OpenCV windows
    if RecordingInfo.showingProcessingWindow:
        cv2.waitKey(1)
        cv2.destroyWindow('Processing') 
    
    return


class RecordingInfo:
    showRecording = True
    showProcessed = True
    stop_video_feed = False
    processed_frame = None
    end = False
    showingRecordingWindow = False
    showingProcessingWindow = False
    capture = False


if __name__ == "__main__":

    # Create and start threads for sending and receiving
    record_thread = threading.Thread(target=ask_recognitions, daemon=True)
    record_thread.start()

    if RecordingInfo.showProcessed:
        show_processed_thread = threading.Thread(target=show_processed_feed, daemon=True)
        show_processed_thread.start()

    # Wait for the threads to finish
    # record_thread.join()
    # if RecordingInfo.showProcessed:
    #     show_processed_thread.join()

    # To allow Ctrl+C
    while record_thread.is_alive() or show_processed_thread.is_alive():
        sleep(100)
    print('acabat')
