# Islamic Prayer Recognition Using YOLOv8

## Overview

This project focuses on detecting and recognizing Islamic prayer actions using a custom-trained YOLOv8 model. The goal is to accurately identify specific prayer postures such as Ruku (bowing), Sujood (prostration), and standing in real-time video streams. The application is designed to process video frames, annotate detected actions with bounding boxes, and display relevant information like action type and confidence level.

## Features

- **Real-Time Detection**: Processes video frames in real-time to detect Islamic prayer actions.
- **Custom YOLOv8 Model**: Utilizes a YOLOv8 model fine-tuned on a dataset specifically designed for recognizing prayer postures.
- **Dynamic Annotations**: Displays bounding boxes and text labels on detected actions, with adjustable background and border styling.
- **User-Friendly Interface**: Outputs annotated video with clear, informative visuals.

## Installation

### Prerequisites

- Python 3.8 or later
- `pip` package manager

### Required Libraries

Install the required Python libraries using the following command:

```bash
pip install ultralytics opencv-python
```

### Clone the Repository

```bash
git clone https://github.com/MohammedHamza0/IslamicPrayerRecognition.git
cd IslamicPrayerRecognition
```

### Model File

Ensure that the `IslamicBest.pt` model file is placed in the project directory. If not, update the path accordingly in the code.

## Usage

1. **Change Directory**: 
   Navigate to the directory containing your project files.

   ```python
   os.chdir(r"F:\YOLO Projects\IslamicPray")
   ```

2. **Load the Model**:
   The model is loaded using the `YOLO` class from the `ultralytics` library.

   ```python
   model = YOLO("IslamicBest.pt")
   ```

3. **Run the Script**:
   Run the script to start processing the video.

   ```bash
   python islamic_prayer_recognition.py
   ```

4. **Video Input**:
   The script processes the video specified in the `cv2.VideoCapture` method. Replace `"istockphoto-1345393460-640_adpp_is.mp4"` with the path to your video file.

5. **Real-Time Display**:
   The processed video frames are displayed in a window titled "IslamicPray". The detection results, including action type and confidence score, are overlaid on the video.

6. **Exit**:
   Press the `Esc` key to exit the video display window.

## Functionality

### `draw_text_with_background()`
This function overlays text with a background and border on the video frames for better visualization.

### Main Script
The script captures video frames, passes them through the YOLOv8 model for action detection, and annotates the frames with bounding boxes and text labels.

