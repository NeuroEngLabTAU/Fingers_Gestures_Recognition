# Fingers Gestures Recognition
## Hand Gesture Prediction using sEMG and Ultraleap Motion Controller 2

This repository contains the implementation of a machine learning project aimed at predicting hand gestures based on surface electromyography (sEMG) signals using a 16-electrode array from Xtrodes Ltd. The Ultraleap Motion Controller 2 is used as the "ground truth" for finger gestures. The project includes scripts for data collection, model training, and evaluation.

## Table of Contents
- [Project Overview](#project-overview)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
  - [Usage](#usage)
- [Data Collection](#data-collection)
  - [Hardware Setup](#hardware-setup)
  - [Data Acquisition](#data-acquisition)
  - [Data Synchronization](#data-synchronization)
- [Model Training and Evaluation](#model-training-and-evaluation)
  - [Preprocessing](#preprocessing)
  - [Model Training](#model-training)
  - [Model Evaluation](#model-evaluation)
- [Results](#results)
- [Repository Structure](#repository-structure)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgements](#acknowledgements)

## Project Overview
This project aims to create a model that accurately predicts hand gestures from sEMG data collected via Xtrodes Ltd electrodes. The Ultraleap Motion Controller 2 is used as the reference for actual finger movements, allowing for supervised learning of the gesture prediction model.

*Include a brief description of the project's significance, the challenges addressed, and potential applications.*

## Getting Started

### Prerequisites
- Python 3.8
- Required Python packages (listed in `requirements.txt`)
- Xtrodes Ltd sEMG electrode array (16 electrodes)
- Ultraleap Motion Controller 2
- Bluetooth-enabled data acquisition unit (DAU) compatible with Xtrodes Ltd electrodes

### Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/hand-gesture-prediction.git

2. Navigate to the project directory:
   ```bash
   cd hand-gesture-prediction

3. Install the required packages:
   ```bash
   pip install -r requirements.txt

## Usage
*Provide instructions on how to run the data collection scripts, train the model, and evaluate the results.*

## Data Collection
### Hardware Setup
*Provide detailed instructions on setting up the Xtrodes Ltd sEMG electrodes and the Ultraleap Motion Controller 2, including connection details and positioning.*

### Data Acquisition
*Explain how to use the scripts to start data collection. Include details about the Bluetooth integration with the DAU and how to initiate simultaneous recording with the Leap Motion Controller 2.*

### Data Synchronization
*Describe how the data from the sEMG electrodes and the Leap Motion Controller 2 are synchronized, including any time-stamping or alignment techniques used.*

## Model Training and Evaluation
### Preprocessing
*Explain the preprocessing steps applied to the raw sEMG data and the corresponding gesture data from the Leap Motion Controller 2. This might include filtering, normalization, feature extraction, etc.*

### Model Training
*Provide an overview of the machine learning models used (e.g., RandomForest, XGBoost) and how to train these models using the preprocessed data.*

### Model Evaluation
*Describe the evaluation metrics used to assess model performance (e.g., accuracy, F1 score) and provide examples of how to interpret the results.*

## Results
*Summarize the results of the model training and evaluation. Include any key findings, visualizations, or insights that demonstrate the effectiveness of the approach.*

## Repository Structure
   ```bash
   hand-gesture-prediction/
   │
   ├── data/                # Raw and processed data files
   ├── notebooks/           # Jupyter notebooks for exploration and analysis
   ├── scripts/             # Python scripts for data collection, training, and evaluation
   ├── models/              # Saved models
   ├── results/             # Results of model evaluation and analysis
   ├── requirements.txt     # Python package dependencies
   └── README.md            # Project documentation
```

*Provide additional details about the directory structure if necessary.*

## Contributing
*Guidelines for contributing to the project, including how to report issues and submit pull requests.*

## License
*Specify the license under which the project is distributed.*

## Acknowledgements
*Mention any funding sources, collaborators, or external resources that were instrumental in the completion of the project.*
