import datetime
import os
import cv2
import threading
import time


class VideoRecorder:
    def __init__(self, webcam, fourcc, saving_path):
        self.webcam = webcam
        self.fourcc = fourcc
        self.saving_path = saving_path
        self.stop_event = threading.Event()
        self.recording_thread = None
        self.output_filename = ""

    def record_video(self):
        self.output_filename = os.path.join(self.saving_path, f"video_{datetime.datetime.now().strftime('%Y-%m-%d %H-%M-%S-%f')[:-3]}.avi")
        print(f"Video recording started at {datetime.datetime.now().strftime('%Y-%m-%d %H-%M-%S-%f')[:-3]} ...")

        out = cv2.VideoWriter(self.output_filename, self.fourcc, 30.0, (640, 480))

        while not self.stop_event.is_set():
            ret, frame = self.webcam.read()

            if ret:
                out.write(frame)
            else:
                print("Failed to capture frame")
                break

        print("Video recording stopped at: ", datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3])
        out.release()

    def start_video(self):
        if not self.stop_event.is_set():
            self.stop_event.clear()
            self.recording_thread = threading.Thread(target=self.record_video)
            self.recording_thread.start()
            print("Recording started")

    def stop_video(self, new_saving_path=None):
        if not self.stop_event.is_set():
            self.stop_event.set()
            print("Recording stopped")


