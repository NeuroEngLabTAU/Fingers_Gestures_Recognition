
import os
import sys

# Ensure the directory containing LeapPython.pyd is in the system path
leap_python_path = r'C:\Users\YH006_new\Desktop\Dvir_leap\finger_pose_estimation\data_aquisition\Leap\resources\Windows'
if leap_python_path not in sys.path:
    sys.path.append(leap_python_path)
os.environ['PATH'] += os.pathsep + leap_python_path
# Add the directory containing LeapPython.pyd to the system path
leap_python_path = r'C:\Users\YH006_new\Desktop\Dvir_leap\finger_pose_estimation\data_aquisition\Leap\resources\Windows'
if leap_python_path not in sys.path:
    sys.path.append(leap_python_path)
os.environ['PATH'] += os.pathsep + leap_python_path