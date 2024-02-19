from flask import Flask
from waitress import serve
from gevent.pywsgi import WSGIServer

from flask_socketio import emit, SocketIO
import services
from time import sleep
import threading
from multiprocessing import Process, Pipe

app = Flask(__name__)
sio = SocketIO(app, namespaces='/videoStream')
port = 8080
rec: services.CurrentRecognizer = services.CurrentRecognizer()

MODES = ['plot', 'text']

@app.route("/")
def home():
    return "Hello, this is Robot Vision, a Python microservice offering computer vision functionalities."


@sio.on("connect")
def test_connect():
    print("Connected")
    emit("my response", {"data": "Connected"})


@sio.on('sendFrame', namespace='/videoStream')
def receive_frame(img_64_in, task, method, mode='plot'):

    print('Frame received. Task: '+task+'. Method:'+method)

    img = services.base64_to_image(img_64_in)

    # img_64_out = services.image_to_base64(img)
    # emit('sendPlot', (img_64_out))

    # Update recognizer
    mustUpdate = rec.mustUpdate(task, method)
    if mustUpdate:
        print('-------- Update Recognizer ----------')
        update_process = Process(target=rec.updateRecognizer, daemon=True)
        update_process.start()
        # services.CurrentRecognizer.update(task, method)

    # Get frame from the background
    if len(rec.output_results) > 0:
        result, plot_img = rec.output_results[0]
    else:
        result = None
        plot_img = img

    print('Result:', rec.recognizer)

    if mode == 'plot':
        img_64_out = services.image_to_base64(plot_img)
        emit('sendPlot', (img_64_out))
    elif mode == 'text':
        emit('sendText', (result))
    
# # Background process to run intensive tasks
# b_process = threading.Thread(target=services.process_images, daemon=True)
# b_process.start()

if __name__ == "__main__":
    print('Running server at http://localhost:'+str(port))

    # # Background process to run intensive tasks
    # b_process = threading.Thread(target=services.process_images, daemon=True)
    # b_process.start()

    # Background process to run intensive tasks
    
    b_process = Process(target=services.process_images, kwargs={'rec': rec}, daemon=True)
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