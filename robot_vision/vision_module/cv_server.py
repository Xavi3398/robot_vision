import os
from flask import Flask, request, Response
from gevent.pywsgi import WSGIServer
import cv2
import numpy as np
import tempfile

from flask_socketio import emit, SocketIO
from threading import Lock
from multiprocessing import Process, Pipe, Queue
from multiprocessing.connection import Connection

from robot_vision.vision_module import services

app = Flask(__name__)
sio = SocketIO(app, namespaces='/videoStream', ping_timeout=600, ping_interval=30)
ip = '127.0.0.1'
port = 8080

MODES = ['plot', 'text']

task: str
method: str
expl_method: str
input_frames: Queue
output_results: Queue
parent_conn: Connection
lock: Lock = Lock()

@app.route("/")
def home():
    return "Hello, this is Robot Vision, a Python microservice offering computer vision functionalities."


@app.route("/explainImage", methods=['POST'])
def explainImage():
    
    print('Image received for explanation.')

    new_task = request.args.get('task', type = str)
    new_method = request.args.get('method', type = str)
    new_expl_method = request.args.get('expl_method', default='lime', type = str)

    if new_task is None or new_method is None or new_expl_method is None:
        print('No task, method or expl_method specified!')
        return 'bad request!', 400

    img_64_in = request.data
    img = services.base64_to_image(img_64_in)

    # Update recognizer
    update_recognizer(new_task, new_method, 'explanation', new_expl_method)

    # Put the incoming frame in the queue to be processed
    put_image_in_queue(img)

    # Get frame from the background, blocking.
    _, plot_img = output_results.get()

    if plot_img is None:
        print('Error in explanation!')
        return 'error in explanation!', 500

    img_64_out = services.image_to_base64(plot_img)
    return img_64_out


@app.route("/sendImage", methods=['POST'])
def sendImage():
    
    print('Image received.')

    new_task = request.args.get('task', type = str)
    new_method = request.args.get('method', type = str)
    mode = request.args.get('mode', default='text', type = str)

    if new_task is None or new_method is None:
        print('No task or method specified!')
        return 'bad request!', 400

    img_64_in = request.data
    img = services.base64_to_image(img_64_in)

    # Update recognizer
    update_recognizer(new_task, new_method, mode, None)

    # Put the incoming frame in the queue to be processed
    put_image_in_queue(img)

    # Get frame from the background, blocking.
    result, plot_img = output_results.get()

    if result is None:
        print('Error in recognition!')
        return 'error in recognition!', 500

    if mode == 'plot':
        img_64_out = services.image_to_base64(plot_img)
        return img_64_out
    elif mode == 'text':
        return str(result)

# Receive video and return processed video
@app.route("/sendVideo", methods=['POST'])
def sendVideo():
    
    print('Video received.')

    new_task = request.args.get('task', type = str)
    new_method = request.args.get('method', type = str)
    mode = request.args.get('mode', default='text', type = str)
    step = request.args.get('step', default=1, type = int)

    if new_task is None or new_method is None:
        print('No task or method specified!')
        return 'bad request!', 400

    # Update recognizer
    update_recognizer(new_task, new_method, mode, None)

    # Save video to a temporary file
    file = request.files['file']
    tmpf_in = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
    file.save(tmpf_in.name)

    # Read video
    cap = cv2.VideoCapture(tmpf_in.name, cv2.CAP_ANY)
    frame_i = 0
    results = []
    step_aux = 0

    # Init output if mode is plot
    if mode == 'plot':
        out_video = cv2.VideoWriter('tmp/temp_video.mp4', int(cap.get(cv2.CAP_PROP_FOURCC)), cap.get(cv2.CAP_PROP_FPS), (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))

    # Read input video
    while cap.isOpened():
            
        # Read frame
        ret, frame = cap.read()
    
        # Finish
        if not ret:
            break
        
        if step_aux == 0:
            
            # Put the incoming frame in the queue to be processed
            put_image_in_queue(frame)

            # Get frame from the background, blocking.
            result, plot_img = output_results.get()
        
        step_aux = (step_aux + 1) % step

        # Save result and plot
        results.append(result)
        if mode == 'plot':
            if result is not None:
                out_video.write(plot_img)
            else:
                out_video.write(frame)
        
        frame_i += 1

    cap.release()
    tmpf_in.close()
    os.unlink(tmpf_in.name)

    if mode == 'plot':
        out_video.release()

    if result is None:
        print('Error in recognition!')
        return 'error in recognition!', 500

    if mode == 'plot':
        print('Returning video.')
        return Response(open('tmp/temp_video.mp4', 'rb'), mimetype='video/mp4')
    elif mode == 'text':
        print('Returning text.')
        return str(results)


@sio.on("connect")
def test_connect():
    print("Connected")
    emit("my response", {"data": "Connected"})


@sio.on('sendFrame', namespace='/videoStream')
def receive_frame(img_64_in, new_task, new_method, mode='plot'):

    print('Frame received.')

    img = services.base64_to_image(img_64_in)

    # Update recognizer
    update_recognizer(new_task, new_method, mode, None)

    # Put the incoming frame in the queue to be processed
    put_image_in_queue(img)

    # Get frame from the background without blocking.
    # We use a lock because it may be executed by different threads simultaneously
    lock.acquire()
    if not output_results.empty():
        result, plot_img = output_results.get()
    else:
        result = None
        plot_img = None
    lock.release()

    # If there are pending results, emit them
    if result is not None:
        if mode == 'plot':
            img_64_out = services.image_to_base64(plot_img)
            emit('sendPlot', (img_64_out))
        elif mode == 'text':
            emit('sendText', (str(result)))


def update_recognizer(new_task, new_method, mode, new_expl_method):

    # Update recognizer. We use a lock because receive_frame may be executed by different threads simultaneously
    # and we only want one thread in here
    lock.acquire()
    global task
    global method
    global expl_method
    if task != new_task or method != new_method or expl_method != new_expl_method:
        print('New task: '+new_task+'. New method:'+new_method)
        task = new_task
        method = new_method
        expl_method = new_expl_method
        parent_conn.send([task, method, mode, expl_method])
    lock.release()


def put_image_in_queue(img):
    # Put the incoming frame in the queue to be processed
    # Remove first element if queue is full. No wait so as to not block in case that between full() and get() someone removed all elements
    # We use a lock because it may be executed by different threads simultaneously
    lock.acquire()
    if input_frames.full():
        try:
            input_frames.get_nowait()
        except:
            pass
    try:
        input_frames.put_nowait(img)
    except:
        pass
    lock.release()

if __name__ == "__main__":
    print('Running server at http://'+ip+':'+str(port))

    # Background process to run intensive tasks
    task = None
    method = None
    parent_conn, child_conn = Pipe()
    input_frames: Queue = Queue(1)
    output_results: Queue = Queue(1)
    b_process = Process(target=services.process_images, args=(child_conn, input_frames, output_results), daemon=True)
    b_process.start()

    # gevent server
    http_server = WSGIServer((ip, port), app)
    http_server.serve_forever()
