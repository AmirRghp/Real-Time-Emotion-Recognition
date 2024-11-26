# Real-Time Emotion Recognition App

This application is a **real-time emotion recognition system** built using **PyQt5**, **PyTorch**, and **OpenCV**. The app uses a pre-trained Vision Transformer (ViT) model to predict emotions based on facial expressions captured from a webcam.

---

## Features

- **Real-Time Video Feed**: Displays a live video feed from the webcam.
- **Emotion Detection**: Detects and classifies emotions into categories like Angry, Disgust, Fear, Happy, Sad, Surprise, and Neutral.
- **User-Friendly Interface**: Minimal and modern UI created using PyQt5.
- **Frame Optimization**: Processes every third frame for smoother performance.

---

## Prerequisites

### 1. Install Dependencies
The following Python packages are required:

- `PyQt5`
- `torch`
- `torchvision`
- `transformers`
- `opencv-python`
- `Pillow`

Install all dependencies using pip:
```bash
   pip install -r requirements.txt
```

### 2. Pre-trained Model
Ensure you have a pre-trained PyTorch model (`your-model.pt`) for emotion classification. Update the `self.model` path in the code to point to your model file.

### 3. Haar Cascade for Face Detection
Make sure the `haarcascade_frontalface_default.xml` file is available in OpenCV's data directory. This file is used for detecting faces in the video feed.

---

## Getting Started

### Clone the Repository
```bash
git clone https://github.com/AmirRghp/Real-Time-Emotion-Recognition.git
cd Real-Time-Emotion-Recognition
```

### Run the Application
1. Make sure your webcam is connected and working.
2. Start the app by running:
   ```bash
   python main.py
   ```

---

## Application Usage

### Buttons:
- **Start**: Starts the webcam feed and emotion recognition.
- **Stop**: Stops the webcam feed.
- **Minimize**: Minimizes the application window.
- **Close**: Closes the application.

### Dragging the Window:
- Click and hold the left mouse button anywhere on the window to drag it.

---

## How It Works

1. **Webcam Integration**:
   - Captures frames from the webcam using OpenCV.

2. **Face Detection**:
   - Detects faces in the frame using OpenCV's Haar Cascade classifier.

3. **Emotion Recognition**:
   - Extracts face regions, preprocesses them, and passes them through a Vision Transformer (ViT) model.
   - The model predicts one of seven emotions: Angry, Disgust, Fear, Happy, Sad, Surprise, Neutral.

4. **Real-Time Display**:
   - Draws bounding boxes and emotion labels on detected faces.

---

## File Structure

```
.
├── main.py            # Main application code
├── UI.ui              # PyQt5 Designer file for the UI
├── icons/             # Resource folder for icons
├── your-model.pt      # Pre-trained emotion classification model
├── requirements.txt   # Dependencies list
```

---

## Customization

- **Model**: Replace `your-model.pt` with your trained PyTorch model.
   - i used this data set :
     ```
     https://www.kaggle.com/code/youssefismail20/human-emotion-detection
     ``` 
- **Emotion Labels**: Update the `self.emotions` list in `main.py` to match your model's classes.
- **Camera Input**: Adjust `cv2.VideoCapture(0)` in the `start_video` method for your camera setup (e.g., use `1` for DroidCam).

---

## Troubleshooting

### Common Errors:

1. **Camera Not Detected**:
   - Ensure the camera is properly connected or adjust the index in `cv2.VideoCapture()`.

2. **Model File Issues**:
   - Verify the model file path and compatibility. Ensure the `.pt` file is correctly formatted for PyTorch.

3. **Missing Dependencies**:
   - Ensure all dependencies listed in `requirements.txt` are installed.

---

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.

---

## Contributing

Feel free to submit issues or pull requests to improve this project. Contributions are always welcome!
