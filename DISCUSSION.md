## Overview
This project presents a flexible protocol for hand gesture recognition using sEMG signals and the Ultraleap Motion Controller 2. It focuses on improving the fidelity of gesture recognition by combining visual and sEMG data. The flexibility of this approach allows for more dynamic and real-world applications in areas such as neuroprosthetics, rehabilitation, and human-computer interaction (HCI).

### Key Features:

- Integration of sEMG with hand tracking to enable robust gesture recognition.
- Option to incorporate an additional webcam to record visual data for further analysis.
- Flexibility in capturing a wide range of hand gestures and positions.
 
##

## Data Collection
The data collection pipeline is designed to record both sEMG signals and hand-tracking data from the Ultraleap system. Accurate gesture capture depends on proper camera alignment, and slight deviations can impact data quality. Be sure to position the hand-tracking camera at the optimal distance and angle relative to the participant’s hand. For additional flexibility, the option to record webcam video data (`video=True` in `data_collection.py`) allows for further analysis beyond what the Ultraleap system provides.

### Best Practices:

- Ensure correct camera alignment for reliable hand gesture capture.
- Familiarize participants with the experiment to avoid "junk" data.
- Monitor participant fatigue and offer breaks to maintain high-quality recordings.

##

## Signal Quality and Preprocessing
sEMG signals can be affected by noise from muscle fatigue, motion artifacts, and environmental interference (such as powerline noise). The system uses band-pass filtering and proper electrode placement to reduce noise and improve signal quality. However, variations due to differences in hand anatomy and strength can still impact the data.

### Troubleshooting and Noise Reduction:

- Use filtering techniques to reduce noise.
- Ensure consistent hand and electrode positioning.
- Consider individual differences when training models on sEMG data.

##

## Machine Learning for Gesture Recognition
The primary challenge in using sEMG for gesture recognition lies in the variability of hand positions. Machine learning models must account for different hand orientations and postures to provide robust recognition. Training the model with data from multiple hand positions will improve its generalizability across subjects and gestures.

### Key Considerations:

- Use machine learning models that account for hand positioning variability.
- Synchronize visual and sEMG data for accurate timing during gesture capture.

##

## System Flexibility and Applications
This protocol’s flexibility in capturing dynamic hand positions and gestures makes it suitable for various real-world applications. The system can be applied in fields like neuroprosthetics, rehabilitation, and HCI, offering advantages such as real-time feedback and low-cost setup compared to other motion capture systems.

### Potential Applications:

- **Neuroprosthetics**: Accurately predict hand gestures from sEMG for real-time control of prosthetic limbs.
- **Rehabilitation**: Tailor motor recovery exercises based on gesture performance and muscle activation.
- **Human-Computer Interaction (HCI)**: Enable natural and intuitive gesture-based control systems.
- **Ergonomics**: Study muscle activity and fatigue during different hand positions to improve workplace design.

##

## Limitations
Despite its flexibility, the system does have some limitations, such as a restricted field of view for the hand-tracking camera, noise from powerline interference, and participant fatigue during long sessions. Addressing these challenges will improve overall data quality and robustness.

### Key Limitations:

- Limited camera field of view.
- Potential noise from nearby electronic devices.
- Fatigue during long recording sessions.

##

## Conclusion
This project presents a flexible and cost-effective method for hand gesture recognition, with applications ranging from neuroprosthetics to ergonomics. While there are some limitations, the system’s adaptability, ease of use, and potential for real-time processing offer significant advantages for further research and development.

