from psychopy import visual, core, event, gui, data, sound
# from psychopy.hardware import keyboard
# from psychopy.constants import (NOT_STARTED, STARTED, PLAYING, PAUSED, STOPPED, FINISHED)
import psychopy as ps
import os
from PIL import Image
import random
from pathlib import Path
import argparse
# from threading import Thread
from multiprocessing import Process
import json

import sys
# Add the parent directory of the current working directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '..')))

# try:
from Leap import LeapRecorder, LeapVisuzalizer
from Leap import NeuroLeap as nl
from Leap.resources.Windows import Leap as LeapController
# except Exception as e:
#     print("Leap Motion SDK not found. Leap Motion data will not be recorded.")
#     print(e)
from streamer import Viz, EmgVisualizer, EMG, Data

# for video recording
from video_recorder import VideoRecorder
import cv2
import datetime


class Experiment:
    def __init__(self, num_repetitions=5, gesture_duration=5, rest_duration=5, gesture_directory=None, record=False, video_flag=False, webcam=None):

        self.color_palette = {
            'background': (0.9, 0.9, 0.9),  # Light gray for background
            'text': (-1, -1, -1),  # Black for text
            'pause_button': (0.4, 0.4, 0.8),  # Blue for pause button
            'stop_button': (0.8, 0.4, 0.4),  # Red for stop button
        }
        
        self.num_repetitions = num_repetitions
        self.gesture_duration = gesture_duration
        self.rest_duration = rest_duration
        self.current_gesture_index = 0
        self.gesture_directory = gesture_directory
        self.record = record
        self.video_flag = video_flag
        self.webcam = webcam
        self.exp_info = {}
        self.exp_num = 0
        self.quit_key = 'q'

        self.data_dir = Path(__file__).parent.parent / 'dataset'

        # makedir if it doesn't exist
        self.data_dir.mkdir(parents=True, exist_ok=True)

        # recording objects
        self.emg_data = None
        self.leap_data = None

    def _init_window(self):

        # Setup window
        try:
        # Setup window
            self.window = visual.Window(
                                size=(1400, 900),
                                fullscr=False,
                                screen=0,
                                allowGUI=True,
                                allowStencil=False,
                                monitor='testMonitor',
                                colorSpace='rgb',
                                color=self.color_palette['background'],
                                blendMode='avg',
                                # useFBO=True, 
                                units='height')
            
                # ... (rest of your initialization code)
            self.exp_info['frameRate'] = self.window.getActualFrameRate()
            self.clock = core.Clock()
            welcome_text = 'A series of images will be shown on screen.\n\n\n' \
                    'Perform the gesture only when\n"Perform gesture"\nis written above the image.\n\n\n' \
                    'Relax your arm between gestures.\n\n\n' \
                    '(Press space when ready.)'
            # Window components
            self.instructions_text = visual.TextStim(self.window, text=welcome_text, color=self.color_palette['text'], height=0.05)
            self.countdown_text = visual.TextStim(self.window, text='', pos=(0, 0.2), color=self.color_palette['text'], height=0.15)
            self.exp_end_text = visual.TextStim(self.window, text='Experiment Complete!', color=self.color_palette['text'])
            self.image_text = visual.TextStim(self.window, text='Perform Gesture', pos=(0.5, -0.45), color=self.color_palette['text'], height=0.05)
            self.running = False
        except Exception as e:
            print(f"Error during window initialization: {e}")
        
        # Experiment setup
        self.gestures, self.prepare_gestures = self.load_gesture_images(self.gesture_directory)
        self.num_completed = {
            i: self.num_repetitions for i in self.gestures.keys()
        }
        self.images = list(self.gestures.keys())
        # choose a random gesture to start with
        start_gesture_index = random.randint(0, len(self.images)-1)
        self.current_image = self.images[start_gesture_index]
        self.num_completed[self.current_image] -= 1

    def collect_participant_info(self):
        info_dialog = gui.Dlg(title='Participant Information')
        info_dialog.addField('Participant ID:')
        info_dialog.addField('Age:')
        info_dialog.addField('Gender:', choices=['Male', 'Female'])
        info_dialog.addField('Session:', choices=['1', '2', '3', '4', '5', '6', '7', '8', '9', '10'])
        info_dialog.addField('Position:', choices=['1', '2', '3', '4', '5'])
        info_dialog.show()
        if info_dialog.OK:
            return {
                'Participant': info_dialog.data["Participant ID:"],
                'Age': info_dialog.data["Age:"],
                'Gender': info_dialog.data["Gender:"],
                'session': info_dialog.data["Session:"],
                'position': info_dialog.data["Position:"]
            }
        else:
            core.quit()
    

    # def resize_and_crop(self, pil_image, target=4):
    #     screen_width, screen_height = 1200, 1000
    #     target_dim = ((screen_width//5)*target, (screen_height//5)*target)
    #     pil_image = pil_image.resize(target_dim)
    #     return pil_image

    def resize_and_crop(self, pil_image, target=4):
        # Base screen dimensions (for target=5 this would be a window of size 1200x900)
        base_screen_width, base_screen_height = 1200, 900

        # Scale the target dimensions
        screen_width = (base_screen_width // 5) * target
        screen_height = (base_screen_height // 5) * target

        # Get the original image size
        original_width, original_height = pil_image.size

        # Calculate the aspect ratios
        original_ratio = original_width / original_height
        screen_ratio = screen_width / screen_height

        # Determine the scaling factor based on aspect ratio to fit without distortion
        if original_ratio > screen_ratio:
            # Fit based on width
            new_width = screen_width
            new_height = int(screen_width / original_ratio)
        else:
            # Fit based on height
            new_height = screen_height
            new_width = int(screen_height * original_ratio)

        # Resize the image
        pil_image = pil_image.resize((new_width, new_height))

        return pil_image


    def load_gesture_images(self, gesture_directory):
        gesture_images = []
        image_names = []
        gestures = {}
        prepare_gestures = {}
        gesture_files = os.listdir(gesture_directory)
        for file_name in gesture_files:
            file_name = file_name.lower()
            if file_name.endswith('.png') or file_name.endswith('.jpg'):
                image_path = os.path.join(gesture_directory, file_name)
                # pil_image = Image.open(image_path)
                original_image = Image.open(image_path)
                pil_image = self.resize_and_crop(original_image)
                prepare_gesture_image = self.resize_and_crop(original_image, 2)  # prepare image for the participant to see before performing the gesture
                image = visual.ImageStim(self.window, image=pil_image)
                prepare_image = visual.ImageStim(self.window, image=prepare_gesture_image, pos=(0, -0.28))
                gestures[file_name.split('.')[0]] = image
                prepare_gestures[file_name.split('.')[0]] = prepare_image
                # gesture_images.append(image)
                # image_names.append(file_name.split('.')[0])
        
        return gestures, prepare_gestures

    def check_quit_key(self):
        if event.getKeys(keyList=[self.quit_key]):
            core.quit()
    
    
    def pause_experiment(self):
        self.running = False
        while True:
            self.instructions_text.text = 'Experiment Paused. Press space to resume or q to quit.'
            self.instructions_text.draw()
            self.window.flip()
            keys = event.waitKeys(keyList=['space', 'q'])
            if 'space' in keys:
                self.running = True
                return
            elif 'q' in keys:
                core.quit()
    
    def stop_experiment(self):
        self.running = False

    def show_countdown(self, duration):
        gesture_image = self.prepare_gestures[self.current_image]
        self.image_text.text = self.current_image.upper()
        for i in range(duration, 0, -1):
            self.countdown_text.text = f"Next gesture in {i} seconds"
            self.countdown_text.draw()
            self.image_text.draw()
            gesture_image.draw()
            self.window.flip()
            core.wait(1)

        # play a sound to indicate the start of the gesture
        tone = sound.Sound(value=1400, secs=0.5, octave=4, stereo=True, volume=0.5)
        # tone.setSound(secs=0.5, octave=4, value=1500)
        tone.play()


    def show_gesture(self):

        gesture_image = self.gestures[self.current_image]
        self.image_text.text = self.current_image.upper()
        # self.window.flip()
        self.trigger(f'start_{self.current_image}_{self.num_repetitions - self.num_completed[self.current_image]}')

        # Create a clock
        countdown_clock = core.Clock()

        # Display the gesture for 5 seconds with a countdown
        for i in range(self.gesture_duration, 0, -1):
            # Reset the clock
            countdown_clock.reset()
            while countdown_clock.getTime() < 1:  # Display each number for 1 second
                # Update the countdown text
                self.image_text.draw()
                gesture_image.draw()
                countdown_text = visual.TextStim(self.window, text=f"Perform gesture for {i} seconds", pos=(0, 0.45), color=self.color_palette['text'], height=0.05)
                countdown_text.draw()
                self.window.flip()

        self.trigger(f'end_{self.current_image}_{self.num_repetitions - self.num_completed[self.current_image]}')

        # play a sound to indicate the end of the gesture
        tone = sound.Sound(value=800, secs=0.5, octave=4, stereo=True, volume=0.4)
        # tone.setSound(secs=0.5, octave=4, value=1000)
        tone.play()
    
    def update_gesture(self):

        if len(self.images) == 0:
            return False
        else:
            # Choose a random gesture that has not been completed enough times
            # self.current_gesture_index = random.randint(0, len(self.gesture_images)-1)
            self.current_image = random.choice(self.images)
            self.num_completed[self.current_image] -= 1
            # Remove the gesture if it has been completed enough times
            if self.num_completed[self.current_image] == 0:
                self.images.remove(self.current_image)

            print(sum(self.num_completed.values()))
            # return fasle if there are no more gestures to display
            return True
        
    def trigger(self, msg, verbose=True):
        
        if self.emg_data is not None:
            self.emg_data.add_annotation(msg)
            if verbose:
                print(f'TRIGGER: {self.emg_data.annotations[-1]}')
        elif verbose:
            print(f'TRIGGER: {msg}')


    def do_experiment(self):

        self._init_window()

        print(f"running experiment with {len(self.images)} gestures")

        # show instructions
        self.instructions_text.draw()

        self.window.flip()
        event.waitKeys(keyList=['space'])

        self.show_countdown(self.rest_duration)  # Display countdown for 5 seconds

        self.running = True
        if (self.record):
            self.exp_info['date'] = data.getDateStr()  # add a simple timestamp
            self.exp_info['expName'] = 'fpe - real time'
            self.exp_info['psychopyVersion'] = '2023.2.3'

            self.data_dir = Path(self.data_dir, self.exp_info['Participant'].rjust(3, '0'), f"S{self.exp_info['session']}")
            self.data_dir.mkdir(parents=True, exist_ok=True)

            with open(Path(self.data_dir, "log.txt"), 'w') as f:
                f.write(f"{self.exp_info}\n")
            # start recording
            self.emg_data.save_as = str(Path(self.data_dir, f"fpe_pos{self.exp_info['position']}_{self.exp_info['Participant'].rjust(3, '0')}_S{self.exp_info['session']}_rep{self.exp_num}_BT.edf"))
            self.leap_data.save_as = str(Path(self.data_dir, f"fpe_pos{self.exp_info['position']}_{self.exp_info['Participant'].rjust(3, '0')}_S{self.exp_info['session']}_rep{self.exp_num}_BT.csv"))
            print(f"Saving data to: {self.emg_data.save_as}")
            
            self.emg_data.start()
            self.leap_data.start()

        while self.running:
            keys = event.getKeys()
            if self.quit_key in keys:
                self.stop_experiment()
                break
            elif 'space' in keys:
                self.pause_experiment()
                continue
            
            self.show_gesture()
            if self.record:
                self.trigger(f'end_{self.current_image}')   
            self.show_countdown(self.rest_duration)  
            # self.do_transition()  
            if not self.update_gesture():
                break
        
        if self.record:
            self.trigger('end_experiment')
            self.emg_data.stop()
            self.leap_data.stop()

            self.emg_data.join()
            self.leap_data.join()

        self.exp_end_text.draw()
        self.window.flip()
        core.wait(3)

        self.window.close()
    
    def do_visualise(self):

         # Visualize data stream in main thread:
        secs = 10             # Time window of plots (in seconds)
        ylim = (-1000, 1000)  # y-limits of plots
        ica = False           # Perform and visualize ICA alongside raw data
        update_interval = 10  # Update plots every X ms
        max_points = 250      # Maximum number of data points to visualize per channel (render speed vs. resolution)


        if self.record:
            emg_viz = Viz(self.emg_data, window_secs=secs, plot_exg=True, plot_imu=False, plot_ica=ica,
                    update_interval_ms=update_interval, ylim_exg=ylim, max_points=250)

            emg_viz.start()

    def visualize_leap(self):
        if self.leap_data is not None:
            leap_viz = LeapVisuzalizer()
            leap_viz.start()
    
    def pre_exp(self, emg_data):
        '''
        Pre experiment setup
        '''
       
        # Visualize data stream in main thread:
        secs = 10             # Time window of plots (in seconds)
        ylim = (-350, 350)  # y-limits of plots
        ica = False           # Perform and visualize ICA alongside raw data
        update_interval = 10  # Update plots every X ms
        max_points = 2500      # Maximum number of data points to visualize per channel (render speed vs. resolution)

        # emg_data.start()

        # plot leap data
        leap_viz = LeapVisuzalizer()
        leap_viz.start()

        # plot emg data
        emg_data.visualize = True
        emg_data.start()
        emg_viz = Viz(emg_data, window_secs=secs, plot_exg=True, plot_imu=False, plot_ica=ica,
                update_interval_ms=update_interval, ylim_exg=ylim, max_points=max_points)

        emg_viz.start()
        print("terminated")
        leap_viz.stop()
        leap_viz.join()


    def run(self,emg_data, leap_data):

        '''
        Main function to run the experiment
        '''
        # setup window
        self._init_window()

        # collect participant info
        self.exp_info = self.collect_participant_info()
        self.emg_data = emg_data
        self.leap_data = leap_data

        # thread = Thread(target=self.do_experiment)
        # thread.start()
        print(f"running experiment with {len(self.images)} gestures")

        # show instructions
        self.instructions_text.draw()

        self.window.flip()
        event.waitKeys(keyList=['space'])

        # self.show_countdown(self.rest_duration)  # Display countdown for 5 seconds

        self.running = True
        if (self.record):
            self.exp_info['date'] = data.getDateStr()  # add a simple timestamp
            self.exp_info['expName'] = 'fpe - real time'
            self.exp_info['psychopyVersion'] = ps.__version__

            self.data_dir = Path(self.data_dir, self.exp_info['Participant'].rjust(3, '0'), f"S{self.exp_info['session']}", f"P{self.exp_info['position']}")
            self.data_dir.mkdir(parents=True, exist_ok=True)

            self.leap_data.data_dir = self.data_dir
            # dump json file
            with open(Path(self.data_dir, "log.json"), 'w') as f:
                json.dump(self.exp_info, f)


            # start recording
            self.emg_data.data_dir = self.data_dir
            self.emg_data.save_as = str(Path(self.data_dir, f"fpe_pos{self.exp_info['position']}_{self.exp_info['Participant'].rjust(3, '0')}_S{self.exp_info['session']}_rep{self.exp_num}_BT.edf"))
            self.emg_data.leap_path = str(Path(self.data_dir, f"fpe_pos{self.exp_info['position']}_{self.exp_info['Participant'].rjust(3, '0')}_S{self.exp_info['session']}_rep{self.exp_num}_BT.csv"))
            self.leap_data.save_as = str(Path(self.data_dir, f"fpe_pos{self.exp_info['position']}_{self.exp_info['Participant'].rjust(3, '0')}_S{self.exp_info['session']}_rep{self.exp_num}_BT_full.csv"))

            if self.video_flag:
                saving_path = self.data_dir  # Define your saving path here

                # Define the video writer object
                fourcc = cv2.VideoWriter_fourcc(*"XVID")

                # Create VideoRecorder instance
                self.video_recorder = VideoRecorder(self.webcam, fourcc, saving_path)

                # Start video recording
                self.video_recorder.start_video()

            print(f"Saving data to: {self.emg_data.save_as}")
            
            self.emg_data.start()
            self.leap_data.start()

        while self.running:

            keys = event.getKeys()
            if self.quit_key in keys:
                self.stop_experiment()
                break
            elif 'space' in keys:
                self.pause_experiment()
                continue

            self.show_countdown(self.rest_duration)  # Display countdown for 5 seconds
            self.show_gesture()
            # self.show_countdown(self.rest_duration)
            # self.do_transition()  
            if not self.update_gesture():
                break
        
        if self.record:
            self.trigger('end_experiment')
            self.emg_data.stop()
            self.leap_data.stop()

            self.emg_data.join()
            self.leap_data.join()

        self.exp_end_text.draw()
        self.window.flip()
        core.wait(2)

        self.window.close()
        
        print("terminated")
        
    def start_processes(self, emg_data, leap_data):

        self.emg_data = emg_data
        self.leap_data = leap_data
        

        vis_process = Process(target=self.do_visualise)
        vis_process.start()

        exp_process = Process(target=self.do_experiment)
        exp_process.start()


        exp_process.join()
        vis_process.join()


def main(args):
    # if usage of video recording is desired:
    # takes about 20 seconds to connect to the webcam at the beginning
    video = False
    # recording one video of the whole experiment

    gesture_dir = './images'
    save_dir = './data'

    # experiment setup
    num_repetitions = 2
    gesture_duration = 2
    rest_duration = 2
    record = True

    # Leap
    desktop = True
    
    host = '127.0.0.1'
    port = 20001
    timeout = 20
    verbose = False

    cap = None  # initialize webcam object

    if video:
        print("connecting to webcam...")
        cap = cv2.VideoCapture(1)
        print("connected to webcam at: ", datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3])

        # Check if the webcam is opened successfully
        if not cap.isOpened():
            print("Failed to open webcam")
            cap.release()  # Release the webcam resource
            sys.exit()

    if record:
        controller = LeapController.Controller()
        if not desktop:
            controller.set_policy_flags(LeapController.POLICY_OPTIMIZE_HMD)
        emg_data = EMG(host_name=host, port=port, LeapClient=controller, timeout_secs=timeout, verbose=verbose, save_as="test.edf")
        # leap_data = LeapRecorder(save_dir)
        leap_data = nl.LeapRecorderBasis(save_dir)
    else:
        emg_data = None
        leap_data = None


    experiment = Experiment(num_repetitions, gesture_duration, rest_duration, gesture_directory=gesture_dir, record=record,
                            video_flag=video, webcam=cap)
    
    if args.vis:
        experiment.pre_exp(emg_data=emg_data)
    else:
        experiment.run(emg_data=emg_data, leap_data=leap_data)

    print("experiment terminated")
    if video:
        print("stopping video recording...")
        experiment.video_recorder.stop_video()


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--vis', action='store_true', help='Visualize data stream')
    argparser.add_argument('--desktop', action='store_true', help='Use desktop mode for leap motion')
    args = argparser.parse_args()

    main(args)
    

