# Fingers Gestures Recognition
# Hand Gesture Prediction using sEMG and Ultraleap Motion Controller 2

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
