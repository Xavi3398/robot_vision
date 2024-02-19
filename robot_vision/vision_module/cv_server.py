from flask import Flask
from waitress import serve
from gevent.pywsgi import WSGIServer

from flask_socketio import emit, SocketIO
from time import sleep
from threading import Lock
from multiprocessing import Process, Pipe, Queue
from multiprocessing.connection import Connection

from robot_vision.vision_module import services

app = Flask(__name__)
sio = SocketIO(app, namespaces='/videoStream')
port = 8080

MODES = ['plot', 'text']

task: str
method: str
input_frames: Queue
output_results: Queue
parent_conn: Connection
lock: Lock = Lock()

@app.route("/")
def home():
    return "Hello, this is Robot Vision, a Python microservice offering computer vision functionalities."


@sio.on("connect")
def test_connect():
    print("Connected")
    emit("my response", {"data": "Connected"})


@sio.on('sendFrame', namespace='/videoStream')
def receive_frame(img_64_in, new_task, new_method, mode='plot'):

    print('Frame received.')

    img = services.base64_to_image(img_64_in)

    # Update recognizer. We use a lock because receive_frame may be executed by different threads simultaneously
    # and we only want one thread in here
    lock.acquire()
    global task
    global method
    if task != new_task or method != new_method:
        print('New task: '+new_task+'. New method:'+new_method)
        task = new_task
        method = new_method
        parent_conn.send([task, method, mode=='plot'])
    lock.release()

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

    # Get frame from the background.
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

if __name__ == "__main__":
    print('Running server at http://localhost:'+str(port))

    # Background process to run intensive tasks
    task = None
    method = None
    parent_conn, child_conn = Pipe()
    input_frames: Queue = Queue(10)
    output_results: Queue = Queue(10)
    b_process = Process(target=services.process_images, args=(child_conn, input_frames, output_results), daemon=True)
    b_process.start()

    # gevent server
    http_server = WSGIServer(('', port), app)
    http_server.serve_forever()





# import os
# import requests
# from flask import Flask, jsonify
# from gevent.pywsgi import WSGIServer

# app = Flask(__name__)
# port = 8080

# with app.app_context():
#     import routes
    
# if __name__ == "__main__":
#     print('Running server at http://localhost:'+str(port))

#     # Flask server
#     # app.run(debug=False, host="0.0.0.0", port=str(port))

#     # gevent server
#     http_server = WSGIServer(('', port), app)
#     http_server.serve_forever()

#     # Waitress server
#     # serve(app, host="0.0.0.0", port=port)