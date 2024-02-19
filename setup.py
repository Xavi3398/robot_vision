from setuptools import setup, find_packages

setup(
    name='robot_vision',
    version='1.0',
    packages=find_packages(include=['robot_vision', 'robot_vision.recognition', 'robot_vision.utils', 'robot_vision.cv_module'])
)