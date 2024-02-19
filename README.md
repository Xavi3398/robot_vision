# Robot Vision

Módulo de visión para cualquier robot.

Ver robot_vision/notebooks/Module Testing.ipynb para ver ejemplos de cómo usar las herramientas de visión.


## Instalación

### Instalar paquetes y robot_vision
Es necesario instalar CUDA 11.8 para usar la GPU en la inferencia.

    git clone https://github.com/Xavi3398/robot_vision
    cd robot_vision
    pip install -r requirements1.txt
    pip install -r requirements2.txt
    pip install -r requirements3.txt
    pip install -e .

### Añadir subcarpetas
Para los pesos de los modelos hay que crear una carpeta "models" dentro de robot_vision/robot_vision, y ponerlos ahí.
Para la tarea de reconocimiento facial hay que crear una carpeta "user_faces" dentro de robot_vision/robot_vision, y poner ahí las imágenes de los usuarios a reconocer. 

### Arreglar errores de paquetes específicos
Hay que cambiar dos líneas del código fuente de MiVOLO:
* robot_vision/src/mivolo/mivolo/structures.py -> comentar línea 14: # os.unsetenv("CUBLAS_WORKSPACE_CONFIG")
* robot_vision/src/mivolo/mivolo/models/yolo_detector.py -> comentar línea 12: # os.unsetenv("CUBLAS_WORKSPACE_CONFIG")

Hay que añadir manualmente los pesos de SPIGA porque no se descargan bien automáticamente:
* Copiar robot_vision/robot_vision/models/spiga_wflw.pt en robot_vision/src/spiga/spiga/models/weights


## Instrucciones para usar el módulo de visión en tiempo real

1. Ir a carpeta robot_vision/vision_module.
2. Ejecutar 'python cv_server.py' para iniciar el servidor.
3. Ejecutar (en otra terminal) 'python client.py' para iniciar un cliente de prueba, que graba vídeo desde la webcam y lo envía al servidor para que lo procese. Se puede elegir entre diferentes tareas y métodos. Para parar reconocimiento y cambiar de tarea/método hay que pulsar 'q' con el focus puesto en la ventana de la cámara (no en la terminal), y luego escribir 'y' en la terminal para continuar o 'n' para parar.

Para parar tanto el servidor como el cliente es necesario usar Ctrl+c repetidas veces en las consolas de comandos, debido a la ejecución multiproceso. Esto se arreglará en una versión futura.
