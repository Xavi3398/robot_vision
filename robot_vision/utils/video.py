
import os    
from tqdm import tqdm
import numpy as np
from decord import VideoReader
from decord import cpu, gpu
import cv2
import copy

from robot_vision.recognition.recognizer import Recognizer

class VideoRecognition:
    def __init__(self, recognizer, step) -> None:
        self.recognizer = recognizer
        self.step = step
        pass

    def run(self):
        pass
    
    def get_results_list(self):
        """Get recognition results as a list.
        """
        return self.results

class VideoRecognitionPath(VideoRecognition):
    
    def __init__(self, video_path, recognizer: Recognizer, step=1) -> None:
        """Runs recognition on every frame of the input video. Works with input and output video files.

        Args:
            video_path (string): Video path.
            recognizer (Recognizer): Recognizer to use for each frame.
            step (int, optional): Step to use for input frames. If greater than one, some frames will be skiped,
                for which results from previous frames will be extended. Defaults to 1.
        """
        super().__init__(recognizer, step)
        self.video_path = video_path

    def run(self):
        """Get list of recognition results of a video, each element corresponding to one frame"""
        
        self.results = []
        
        with open(self.video_path, 'rb') as f:
            vr = VideoReader(f, ctx=cpu(0))
            
            for i in tqdm(range(0, len(vr), self.step)):
                frame = vr[i].asnumpy()
                result = self.recognizer.get_result(frame)

                # Extend results to skiped frames
                for _ in range(self.step):
                    self.results.append(result)

    def get_results_plot(self, output_path):
        """Get recognition as an output video.

        Args:
            output_path (string): Path where to write the resulting video.
        """
        
        # Input video
        cap = cv2.VideoCapture(self.video_path, cv2.CAP_ANY)
        frame_i = 0

        # Output video
        # fourcc = cv2.VideoWriter_fourcc(*'MP4V')
        out = cv2.VideoWriter(output_path, int(cap.get(cv2.CAP_PROP_FOURCC)), cap.get(cv2.CAP_PROP_FPS), (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))

        while cap.isOpened():
            
            # Read frame
            ret, frame = cap.read()
        
            # Finish
            if not ret:
                break
            
            # Get corresponding result
            result = self.results[frame_i]

            # Get plot
            out_frame = self.recognizer.get_plot_result(frame, result)

            # Write to output video
            out.write(out_frame)

            frame_i += 1
        cap.release()
        out.release()

class VideoRecognitionNumPy(VideoRecognition):
    
    def __init__(self, video, recognizer: Recognizer, step=1) -> None:
        """Runs recognition on every frame of the input video. Works with input and output NumPy arrays.

        Args:
            video (numpy array): Video.
            recognizer (Recognizer): Recognizer to use for each frame.
            step (int, optional): Step to use for input frames. If greater than one, some frames will be skiped,
                for which results from previous frames will be extended. Defaults to 1.
        """
        super().__init__(recognizer, step)
        self.video = video

    def run(self):
        """Get list of recognition results of a video, each element corresponding to one frame"""
        
        self.results = []
        
        for n_frame in tqdm(range(0, self.video.shape[0], self.step)):
            frame = self.video[n_frame,...]
            result = self.recognizer.get_result(frame)

            # Extend results to skiped frames
            for _ in range(self.step):
                self.results.append(result)

    def get_results_plot(self):
        """Get recognition as an output video.

        Returns:
            numpy array: resulting video.
        """
        
        # Input video
        frame_i = 0

        # Output video
        out_video = copy.deepcopy(self.video)

        for n_frame in tqdm(range(0, self.video.shape[0], self.step)):
            frame = self.video[n_frame,...]
            result = self.results[n_frame]

            # Get plot
            out_frame = self.recognizer.get_plot_result(frame, result)

            # Write to output video
            out_video[n_frame,...] = out_frame
        
        return out_video