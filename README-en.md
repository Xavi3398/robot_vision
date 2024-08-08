# Robot Vision

Leer README en [español](README.md).

Vision module for any robot, presented at the INTERACCION'24 conference: [*"An AI-Powered Computer Vision Module for Social Interactive Agents"*](https://doi.org/10.1145/3657242.3658601).

See [notebooks/Predefined.ipynb](notebooks/Predefined.ipynb) for examples of how to use the vision tools.


## Installation

### Prerequisites

* Windows 10 was used to develop the module. It has not been tested on other operating systems yet.
* For some packages, it is necessary to have the [Microsoft C++ Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/) installed. After downloading the executable, select the "Desktop development with C++" option, and in the installation details, remove the Windows 11 SDK and add the Windows 10 SDK (if using Windows 10).
* [Git](https://git-scm.com/book/en/v2/Getting-Started-Installing-Git) must also be installed.
* Anaconda or [Miniconda](https://docs.anaconda.com/free/miniconda/index.html) is also required to install the module within a new environment.

### Install packages and robot_vision
It is recommended to create a new environment for the module installation. The Python version used to develop the module is 3.8.18. CUDA 11.8 must also be installed to use the GPU for inference.

    conda create -n vision-module python=3.8.18
    conda activate vision-module
    git clone https://github.com/Xavi3398/robot_vision
    cd robot_vision
    pip install -r requirements1.txt (or pip install -r requirements1-CPU.txt if not using CUDA)
    pip install -r requirements2.txt
    pip install -r requirements3.txt
    pip install -e .

### Add subfolders
For the model weights, create a "models" folder inside robot_vision/robot_vision, and put them there.
For the facial recognition task, create a "user_faces" folder inside robot_vision/robot_vision, and put the user images to be recognized there.

### Fix specific package errors
Two lines of MiVOLO source code need to be changed:
* robot_vision/src/mivolo/mivolo/structures.py -> comment out line 14: # os.unsetenv("CUBLAS_WORKSPACE_CONFIG")
* robot_vision/src/mivolo/mivolo/model/yolo_detector.py -> comment out line 12: # os.unsetenv("CUBLAS_WORKSPACE_CONFIG")

SPIGA weights need to be added manually because they do not download automatically:
* Copy robot_vision/robot_vision/models/spiga_wflw.pt to robot_vision/src/spiga/spiga/models/weights

If not using CUDA, remove its usage in the SPIGA package (file src/spiga/spiga/inference/framework.py):
* Comment out line 45: self.model = self.model.cuda(gpus[0])
* Change line 136: data_var = data.cuda(device=self.gpus[0], non_blocking=True) to data_var = data

If not using CUDA, also change the MiVOLO call:
* In file robot_vision/recognition/predefined.py, function predefined_age_gender_MiVOLO(), add device='cpu' to the age_gender.MiVOLOAgeGender call.

## Instructions for using the vision module

### Start the module along with the test client program
1. Go to the robot_vision/vision_module folder.
2. Run 'python cv_server.py' to start the server.
3. Run (in another terminal) 'python client.py' to start a test client, which records video from the webcam and sends it to the server for processing. You can choose between different tasks and methods. To stop recognition and change the task/method, right-click or press 'q' with the focus on the camera window (not the terminal), then type 'y' in the terminal to continue or 'n' to stop.
4. If image mode is selected instead of video mode, only screenshots are processed, not the entire video. To take a screenshot, left-click in the recording window.

To stop both the server and the client, use Ctrl+c repeatedly in the command consoles due to multiprocess execution. This will be fixed in a future version.

### Requests:
The module, as configured, responds to requests at http://127.0.0.1:8080 (localhost, port 8080). It accepts two types of requests:

 - POST of an image (image mode).
 - Messages via sockets with frames to be processed (video mode).

#### Image mode

POST request to http://127.0.0.1:8080/sendImage. This mode is designed to process images on demand (not in real-time). For example, take a screenshot at a specific moment and process it to identify the user in front of the camera.

Named parameters:

 - **task**: task to be performed. Available: *'face_detection', 'person_detection', 'keypoints', 'expression', 'age_gender', 'face_recognition', 'background_subtraction'*.
 - **method**: method to use. Depends on the task. See PREDEFINED_RECOGNIZERS dictionary in robot_vision/recognition/predefined.py for the full list.
 - **mode**: type of response for the request. Available: *'text'* (returns text with the recognition result), *'plot'* (returns the input image with the result overlaid).

POST body: the image to process, in bytes, compressed in jpg and base 64.

#### Real-time video mode

Sending messages via sockets to http://127.0.0.1:8080/videoStream/sendFrame (namespace='/videoStream'). Designed to send real-time video. Parameters (unnamed):

- the image to process, in bytes, compressed in jpg and base 64.
- task to be performed. Available: *'face_detection', 'person_detection', 'keypoints', 'expression', 'age_gender', 'face_recognition', 'background_subtraction'*.
- method to use. Depends on the task. See PREDEFINED_RECOGNIZERS dictionary in robot_vision/recognition/predefined.py for the full list.
- type of response for the request. Available: *'text'* (returns text with the recognition result), *'plot'* (returns the input image with the result overlaid).

Recognition results are emitted, also via sockets, at:

- http://CLIENT_IP:CLIENT_PORT/videoStream/sendPlot if the mode is *'plot'*.
- http://CLIENT_IP:CLIENT_PORT/videoStream/sendText if the mode is *'text'*.

## Available tasks and models

| **Task**                   | **Year** | **Method**             | **Speed (imgs/s)** | **Accuracy**  | **Additional notes**                                                                    | **Implementation**                                              |
|------------------------|------|--------------------|----------------|-----------|-------------------------------------------------------------------------------------|-------------------------------------------------------------|
| Face detection         | 2023 | YOLOv8             | 85.51          | Very Good |                                                                                     | [Official code](https://github.com/ultralytics/ultralytics) |
|                        | 2021 | SCRFD              | 547.38         | Very Good |                                                                                     | [InsightFace](https://github.com/deepinsight/insightface)   |
|                        | 2016 | MTCNN              | 1.69           | Very Good |                                                                                     | [TensorFlow](https://github.com/ipazc/mtcnn)                |
|                        | 2005 | HOG + SVM          | 7.97           | Good      |                                                                                     | [DLIB](http://dlib.net/python)                              |
|                        | 2001 | Viola-Jones (face) | 5.75           | Fair      | Requires good choice of hyperparameters to get good results.                        | [OpenCV](https://docs.opencv.org/3.4)                       |
| Face recognition       | 2015 | ResNet50           | 15.73          | Very Good |                                                                                     | [InsightFace](https://github.com/deepinsight/insightface)   |
| Person detection       | 2023 | YOLOv8             | 76.24          | Very Good |                                                                                     | [Official code](https://github.com/ultralytics/ultralytics) |
| Facial landmarks       | 2022 | SPIGA              | 11.19          | Very Good | **98 landmarks**.                                                                       | [Official code](https://github.com/andresprados/SPIGA)      |
|                        | 2017 | MobileNet          | 122.87         | Very Good | **106 landmarks**.                                                                      | [InsightFace](https://github.com/deepinsight/insightface)   |
|                        | 2016 | MTCNN              | 0.82           | Good      | **5 landmarks**: eye_left, eye_right, nose, mouth_left, mouth_right.                    | [TensorFlow](https://github.com/ipazc/mtcnn)                |
|                        | 2014 | Regression Trees   | 56.88          | Good      | **68 landmarks**.                                                                       | [DLIB](http://dlib.net/python)                              |
|                        | 2001 | Viola-Jones (eyes) | 10.19          | Fair      | **2 landmarks**: the eyes. Requires good choice of hyperparameters to get good results. | [OpenCV](https://docs.opencv.org/3.4)                       |
| Age and gender         | 2023 | MiVOLO             | 10.78          | Very Good

 |                                                                                     | [Official code](https://github.com/WildChlamydia/MiVOLO)    |
|                        | 2017 | MobileNet          | 91.73          | Good      | Does not work for children.                                                         | [InsightFace](https://github.com/deepinsight/insightface)   |
| Expression recognition | 2022 | SilNet             | 5.92           | Very Good |                                                                                     | [Keras](https://keras.io/api/applications)                  |
|                        | 2021 | EfficientNetV2     | 4.99           | Very Good |                                                                                     | [Keras](https://keras.io/api/applications)                  |
|                        | 2019 | MobileNetV3        | 5.60           | Very Good |                                                                                     | [Keras](https://keras.io/api/applications)                  |
|                        | 2017 | Xception           | 5.62           | Very Good |                                                                                     | [Keras](https://keras.io/api/applications)                  |
|                        | 2015 | InceptionV3        | 2.99           | Very Good |                                                                                     | [Keras](https://keras.io/api/applications)                  |
|                        | 2015 | ResNet50           | 4.51           | Very Good |                                                                                     | [Keras](https://keras.io/api/applications)                  |
|                        | 2015 | ResNet101V2        | 3.93           | Very Good |                                                                                     | [Keras](https://keras.io/api/applications)                  |
|                        | 2015 | VGG16              | 4.72           | Very Good |                                                                                     | [Keras](https://keras.io/api/applications)                  |
|                        | 2015 | VGG19              | 4.39           | Very Good | This is the model we have tested the most.                                          | [Keras](https://keras.io/api/applications)                  |
|                        | 2015 | WeiNet             | 6.95           | Very Good |                                                                                     | [Keras](https://keras.io/api/applications)                  |
|                        | 2014 | SongNet            | 6.34           | Very Good |                                                                                     | [Keras](https://keras.io/api/applications)                  |
|                        | 2012 | AlexNet            | 6.60           | Very Good | Good choice if the extra speed is required.                                         | [Keras](https://keras.io/api/applications)                  |
| Background subtraction | 2020 | U2-Net             | 2.53           | Very Good |                                                                                     | [RemBG](https://github.com/danielgatis/rembg)               |

Examples:

![Tasks](./notebooks/resources/tasks.png)

## Explainability

| **Method**            | **Speed** | **_'Quality'_** | **Segmentation** | **Relevance**         |
|-----------------------|-----------|---------------|------------------|-----------------------|
| LIME                  | Medium    | Good          | SLIC superpixels | Positive and Negative |
| SHAP                  | Slow      | Good          | SLIC superpixels | Positive and Negative |
| Kernel SHAP           | Medium    | Good          | SLIC superpixels | Positive and Negative |
| RISE                  | Medium    | Good          | Grid             | Heatmap               |
| Occlusion Sensitivity | Medium    | Medium        | Grid             | Heatmap               |
| LOCO                  | Fast      | Bad           | SLIC superpixels | Positive and Negative |
| Univariate Predictors | Fast      | Very bad      | SLIC superpixels | Positive and Negative |

***Note***:  *For methods that offer positive and negative relevance, positive relevance is shown in green and negative relevance in red. Positive relevance indicates those regions which, when removed, prevent the model from recognizing the class. Negative relevance indicates those regions which, when removed, help the model recognize the class.*

Examples:
![Explainability Methods](./notebooks/resources/explanations.png)

## License
This project is licensed under the terms of the MIT license. See the [LICENSE](LICENSE) file for more details.

## Acknowledgments
This work is part of the Project PID2022-136779OB-C32 (PLEISAR) funded by MICIU/ AEI /10.13039/501100011033/ and FEDER, EU. Project Playful Experiences with Interactive Social Agents and Robots (PLEISAR): Social Learning and Intergenerational Communication.

Grant PID2019-104829RA-I00 funded by MCIN/ AEI /10.13039/501100011033. Project EXPLainable Artificial INtelligence systems for health and well-beING (EXPLAINING).

F. X. Gaya-Morey was supported by an FPU scholarship from the Ministry of European Funds, University and Culture of the Government of the Balearic Islands.

## Citation
If you use this code in your research, please cite our paper:

```
@inproceedings{gaya_morey2024ai,
	title        = {An AI-Powered Computer Vision Module for Social Interactive Agents},
	author       = {Gaya Morey, Francesc Xavier and Manresa-Yee, Cristina and Buades Rubio, Jose Maria},
	year         = 2024,
	booktitle    = {Proceedings of the XXIV International Conference on Human Computer Interaction},
	location     = {A Coru\~{n}a, Spain},
	publisher    = {Association for Computing Machinery},
	address      = {New York, NY, USA},
	series       = {Interacci\'{o}n '24},
	doi          = {10.1145/3657242.3658601},
	isbn         = 9798400717871,
	articleno    = 19,
	numpages     = 5
}
```