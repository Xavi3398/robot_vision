
# Robot Vision

Módulo de visión para cualquier robot.

Ver robot_vision/notebooks/Module Testing.ipynb para ver ejemplos de cómo usar las herramientas de visión.

## Instalación

    git clone https://github.com/Xavi3398/robot_vision
    cd robot_vision
    pip install -r requirements.txt
    pip install -e .

Además, hay que cambiar dos líneas del código fuente de MiVOLO:
* robot_vision/src/mivolo/mivolo/structures.py -> comentar línea 14: # os.unsetenv("CUBLAS_WORKSPACE_CONFIG")
* robot_vision/src/mivolo/mivolo/models/yolo_detector.py -> comentar línea 12: # os.unsetenv("CUBLAS_WORKSPACE_CONFIG")

## Instrucciones para usar el módulo de visión en tiempo real

1. Ir a carpeta robot_vision/vision_module.
2. Ejecutar 'python cv_server.py' para iniciar el servidor.
3. Ejecutar (en otra terminal) 'python client.py' para iniciar un cliente de prueba, que graba vídeo desde la webcam y lo envía al servidor para que lo procese. Se puede elegir entre diferentes tareas y métodos.