
# Robot Vision

Módulo de visión para cualquier robot.

Ver robot_vision/notebooks/Module Testing.ipynb para ver ejemplos de cómo usar las herramientas de visión.


## Instalación

### Prerequisitos

* Para desarrollar el módulo se ha usado Windows 10. Todavía no se ha provado en otros sistemas operativos.
* Para algunos paquetes es necesario tener instaladas las [Microsoft C++ Build Tools](https://visualstudio.microsoft.com/es/visual-cpp-build-tools/). Después de descargar el ejecutable, seleccionar la opción "Desktop development with C++", y en los detalles de instalación quitar Windows 11 SDK y añadir Windows 10 SDK (si se usa Windows 10).
* También es necesario tener instalado [Git](https://git-scm.com/book/en/v2/Getting-Started-Installing-Git).
* Y Anaconda o [Miniconda](https://docs.anaconda.com/free/miniconda/index.html), para instalar el módulo dentro de un nuevo entorno.

### Instalar paquetes y robot_vision
Se recomienda crear un nuevo entorno (environment) para la instalación del módulo. La versión de Python usada para desarrollar el módulo es la 3.8.18. También es necesario instalar CUDA 11.8 para usar la GPU en la inferencia.

    conda create -n vision-module python=3.8.18
    conda activate vision-module
    git clone https://github.com/Xavi3398/robot_vision
    cd robot_vision
    pip install -r requirements1.txt (o bien pip install -r requirements1-CPU.txt si no se usa CUDA)
    pip install -r requirements2.txt
    pip install -r requirements3.txt
    pip install -e .

### Añadir subcarpetas
Para los pesos de los modelos hay que crear una carpeta "models" dentro de robot_vision/robot_vision, y ponerlos ahí.
Para la tarea de reconocimiento facial hay que crear una carpeta "user_faces" dentro de robot_vision/robot_vision, y poner ahí las imágenes de los usuarios a reconocer. 

### Arreglar errores de paquetes específicos
Hay que cambiar dos líneas del código fuente de MiVOLO:
* robot_vision/src/mivolo/mivolo/structures.py -> comentar línea 14: # os.unsetenv("CUBLAS_WORKSPACE_CONFIG")
* robot_vision/src/mivolo/mivolo/model/yolo_detector.py -> comentar línea 12: # os.unsetenv("CUBLAS_WORKSPACE_CONFIG")

Hay que añadir manualmente los pesos de SPIGA porque no se descargan bien automáticamente:
* Copiar robot_vision/robot_vision/models/spiga_wflw.pt en robot_vision/src/spiga/spiga/models/weights

Si no se usa CUDA, hay que eliminar su uso en el paquete de SPIGA (archivo src/spiga/spiga/inference/framework.py):
* Comentar línea 45: self.model = self.model.cuda(gpus[0])
* Cambiar línea 136: data_var = data.cuda(device=self.gpus[0], non_blocking=True) por data_var = data

Si no se usa CUDA, cambiar también llamada a MiVOLO:
* En archivo robot_vision/recognition/predefined.py, función predefined_age_gender_MiVOLO(), añadir device='cpu' a la llamada a age_gender.MiVOLOAgeGender

## Instrucciones para usar el módulo de visión

### Arrancar el módulo, junto con el programa cliente de prueba
1. Ir a carpeta robot_vision/vision_module.
2. Ejecutar 'python cv_server.py' para iniciar el servidor.
3. Ejecutar (en otra terminal) 'python client.py' para iniciar un cliente de prueba, que graba vídeo desde la webcam y lo envía al servidor para que lo procese. Se puede elegir entre diferentes tareas y métodos. Para parar reconocimiento y cambiar de tarea/método hay que pulsar botón derecho del ratón o 'q' con el focus puesto en la ventana de la cámara (no en la terminal), y luego escribir 'y' en la terminal para continuar o 'n' para parar.
4. Si se ha seleccionado el modo imagen en vez de vídeo, sólo se procesan las capturas de pantalla, no todo el vídeo. Para hacer una captura de pantalla, hacer click izquierdo con el ratón en la ventana de la grabación.

Para parar tanto el servidor como el cliente es necesario usar Ctrl+c repetidas veces en las consolas de comandos, debido a la ejecución multiproceso. Esto se arreglará en una versión futura.

### Peticiones:
El módulo, tal como está configurado, contesta peticiones en http://127.0.0.1:8080 (localhost, puerto 8080). Acepta dos clases de peticiones:

 - POST de una imagen (modo imagen).
 - Mensajes a través de sockets con fotogramas a procesar (modo vídeo)

#### Modo imagen

Petición POST a http://127.0.0.1:8080/sendImage. Este modo está pensado para procesar imágenes a petición (no en tiempo real). Por ejemplo, hacer una captura de pantalla en un momento determinado y procesarla para identificar el usuario delante de la cámara.

Parámetros con nombre:

 - **task**: tarea a realizar. Disponibles: *'face_detection', 'person_detection', 'keypoints', 'expression', 'age_gender', 'face_recognition', 'background_subtraction'*.
 - **method**: método a utilizar. Depende de la tarea. Ver diccionario PREDEFINED_RECOGNIZERS en robot_vision/recognition/predefined.py para la lista completa.
 - **mode**: tipo de respuesta de la petición. Disponibles: *'text'* (devuelve texto con el resultado del reconocimiento), *'plot'* (devuelve la imagen de entrada con el resultado pintado encima).

Cuerpo del POST: la imagen a procesar, en bytes, comprimida en jpg y en base 64.

#### Modo vídeo en tiempo real

Envío de mensajes a través de sockets a http://127.0.0.1:8080/videoStream/sendFrame (namespace='/videoStream'). Pensado para enviar vídeo en tiempo real. Parámetros (sin nombre):

- la imagen a procesar, en bytes, comprimida en jpg y en base 64.
- tarea a realizar. Disponibles: *'face_detection', 'person_detection', 'keypoints', 'expression', 'age_gender', 'face_recognition', 'background_subtraction'*.
- método a utilizar. Depende de la tarea. Ver diccionario PREDEFINED_RECOGNIZERS en robot_vision/recognition/predefined.py para la lista completa.
- tipo de respuesta de la petición. Disponibles: *'text'* (devuelve texto con el resultado del reconocimiento), *'plot'* (devuelve la imagen de entrada con el resultado pintado encima).

Los resultados del reconocimiento se emiten, también a través de sockets, en:

- http://IP_CLIENTE:PUERTO_CLIENTE/videoStream/sendPlot si el modo es *'plot'*.
- http://IP_CLIENTE:PUERTO_CLIENTE/videoStream/sendText, si el modo es *'text'*.

## Tareas y modelos disponibles

| Task              | Year          | Method                   | Implementation                                              |
|-------------------|---------------|--------------------------|-------------------------------------------------------------|
| Face detection    | 2023          | YOLOv8                   | [Official code](https://github.com/ultralytics/ultralytics) |
|                   | 2021          | SCRFD                    | [InsightFace](https://github.com/deepinsight/insightface)   |
|                   | 2016          | MTCNN                    | [TensorFlow](https://github.com/ipazc/mtcnn)                |
|                   | 2005          | HOG + SVM                | [DLIB](http://dlib.net/python)                              |
|                   | 2001          | Viola-Jones (face)       | [OpenCV](https://docs.opencv.org/3.4)                       |
| Face recognition  | 2015          | ResNet50                 | [InsightFace](https://github.com/deepinsight/insightface)   |
| Person detection  | 2023          | YOLOv8                   | [Official code](https://github.com/ultralytics/ultralytics) |
| Facial landmarks  | 2022          | SPIGA                    | [Official code](https://github.com/andresprados/SPIGA)      |
|                   | 2017          | MobileNet                | [InsightFace](https://github.com/deepinsight/insightface)   |
|                   | 2016          | MTCNN                    | [TensorFlow](https://github.com/ipazc/mtcnn)                |
|                   | 2014          | Regression Trees         | [DLIB](http://dlib.net/python)                              |
|                   | 2001          | Viola-Jones (eyes)       | [OpenCV](https://docs.opencv.org/3.4)                       |
| Age and gender    | 2023          | MiVOLO                   | [Official code](https://github.com/WildChlamydia/MiVOLO)    |
|                   | 2017          | MobileNet                | [InsightFace](https://github.com/deepinsight/insightface)   |
| Expression recognition   | 2022   | SilNet                   | [Keras](https://keras.io/api/applications)                  |
|                   | 2021          | EfficientNetV2           | [Keras](https://keras.io/api/applications)                  |
|                   | 2019          | MobileNetV3              | [Keras](https://keras.io/api/applications)                  |
|                   | 2017          | Xception                 | [Keras](https://keras.io/api/applications)                  |
|                   | 2015          | InceptionV3              | [Keras](https://keras.io/api/applications)                  |
|                   | 2015          | ResNet50                 | [Keras](https://keras.io/api/applications)                  |
|                   | 2015          | ResNet101V2              | [Keras](https://keras.io/api/applications)                  |
|                   | 2015          | VGG16                    | [Keras](https://keras.io/api/applications)                  |
|                   | 2015          | VGG19                    | [Keras](https://keras.io/api/applications)                  |
|                   | 2015          | WeiNet                   | [Keras](https://keras.io/api/applications)                  |
|                   | 2014          | SongNet                  | [Keras](https://keras.io/api/applications)                  |
|                   | 2012          | AlexNet                  | [Keras](https://keras.io/api/applications)                  |
| Background subtraction | 2020     | U<sup>2</sup>-Net        | [RemBG](https://github.com/danielgatis/rembg)               |
