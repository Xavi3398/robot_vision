from time import sleep
import cv2
import socketio #python-socketio by @miguelgrinberg
import services
import threading

from robot_vision.recognition.predefined import PREDEFINED_RECOGNIZERS

sio = socketio.Client()
sio.connect('http://localhost:8080', namespaces='/videoStream')


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
        print('\nAvailable tasks:')
        print(get_as_options(list(PREDEFINED_RECOGNIZERS.keys())))
        choice = int(input('\nIntroduce a task: ')) - 1
        task = list(PREDEFINED_RECOGNIZERS.keys())[choice if choice > 0 else 0]
        
        print('\nAvailable methods:')
        print(get_as_options(list(PREDEFINED_RECOGNIZERS[task].keys())))
        choice = int(input('\nIntroduce a method: ')) - 1
        method = list(PREDEFINED_RECOGNIZERS[task].keys())[choice if choice > 0 else 0]

        print('\nStarting recording. Press "q" to stop.\n')

        record_and_send_webcam(task, method)

        continue_recognition = ''
        while continue_recognition not in ['y', 'n', 'yes', 'no']:
            continue_recognition = input('Run new recognition (y/n)?').lower()
        
        if continue_recognition not in['y', 'yes']:
            RecordingInfo.end = True


def record_and_send_webcam(task, method):
    RecordingInfo.stop_video_feed = False
    cam = cv2.VideoCapture(0)

    while (not RecordingInfo.stop_video_feed):

        # get frame from webcam
        ret, frame = cam.read()                     

        # Close
        if RecordingInfo.stop_video_feed or not ret or (cv2.waitKey(1) & 0xFF == ord('q')):
            RecordingInfo.stop_video_feed = True
        
        else:
            # Show recording
            if RecordingInfo.showRecording:
                cv2.imshow('Recording', frame) 
            
            # send to server
            img_64_out = services.image_to_base64(frame)
            sio.emit('sendFrame', (img_64_out, task, method, 'plot'), namespace='/videoStream')

    # Release resources
    cam.release()


@sio.on('sendPlot', namespace='/videoStream')
def receive_and_show_plot(img_64_in):
    print('Processed image received.')
    img = services.base64_to_image(img_64_in)
    RecordingInfo.processed_frame = img


@sio.on('sendText', namespace='/videoStream')
def receive_and_show_text(message):
    print('Result received: ' + message)


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
        

class RecordingInfo:
    showRecording = True
    showProcessed = True
    stop_video_feed = False
    processed_frame = None
    end = False


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
        # show_processed_thread.join()

    while record_thread.is_alive() or show_processed_thread.is_alive():
        sleep(100)

    # Close OpenCV windows
    cv2.waitKey(1)
    cv2.destroyAllWindows()   