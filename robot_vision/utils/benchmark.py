import time
import cv2
import os
import numpy as np
from tqdm import tqdm

from robot_vision.recognition.recognizer import Recognizer

class Benchmark:

    def __init__(self, imgs_path, method: Recognizer, max_samples=None) -> None:
        self.imgs_path = imgs_path
        self.method = method
        self.max_samples = max_samples

    def run(self):

        self.times = []
        self.total_time = None

        # Run first time without counting, in case initialization is needed
        img_name = os.listdir(self.imgs_path)[0]
        img = cv2.imread(os.path.join(self.imgs_path, img_name))
        self.method.get_result(img)

        start_time = time.time()

        paths = os.listdir(self.imgs_path) if self.max_samples is None else os.listdir(self.imgs_path)[:self.max_samples]
        for img_name in tqdm(paths):
            img = cv2.imread(os.path.join(self.imgs_path, img_name))

            time1 = time.time()
            self.method.get_result(img)
            self.times.append(time.time()-time1)

        self.total_time = time.time() - start_time
        return
    
    def print_report(self):
        print("Number of imgs: %d" % len(self.times))
        print()
        print("Total time:     %.2f s" % self.get_total_time())
        print("Inference time: %.2f s" % self.get_inference_time())
        print("Read time:      %.2f s" % self.get_read_time())
        print()

        if self.get_avg_total_time() < 1:
            print("Avg. speed:           %.2f imgs/s" % self.get_total_speed())
        else:
            print("Avg. speed:           %.2f s" % self.get_avg_total_time())

        if self.get_avg_inference_time() < 1:
            print("Avg. inference speed: %.2f imgs/s" % self.get_inference_speed())
        else:
            print("Avg. inference speed: %.2f s" % self.get_avg_inference_time())

        if self.get_avg_read_time() < 1:
            print("Avg. read speed:      %.2f imgs/s" % self.get_read_speed())
        else:
            print("Avg. read speed:      %.2f s" % self.get_avg_read_time())
    
    def get_read_time(self):
        return self.get_total_time() - self.get_inference_time()
    
    def get_inference_time(self):
        return np.sum(self.times)
    
    def get_total_time(self):
        return self.total_time
    
    def get_inference_speed(self):
        return 1 / self.get_avg_inference_time()
    
    def get_total_speed(self):
        return 1 / self.get_avg_total_time()
    
    def get_read_speed(self):
        return 1 / self.get_avg_read_time()
    
    def get_avg_inference_time(self):
        return np.mean(self.times)
    
    def get_avg_total_time(self):
        return self.total_time / len(self.times)
    
    def get_avg_read_time(self):
        return self.get_avg_total_time() - self.get_avg_inference_time()