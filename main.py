import sys
import cv2
import torch
import numpy as np
from PyQt5.QtWidgets import QApplication, QMainWindow
from PyQt5.QtCore import QTimer , Qt
from PyQt5.QtGui import QImage, QPixmap
from PyQt5 import uic , QtCore
from torchvision import transforms
from transformers import ViTForImageClassification
import icons.icons_rc


class EmotionRecognition(QMainWindow):
    def __init__(self):
        super(EmotionRecognition, self).__init__()
        uic.loadUi('./UI.ui', self)

        # remove windows title bar
        self.setWindowFlag(Qt.WindowType.FramelessWindowHint)

        # set main background transparent
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)

        self.timer = QTimer()
        self.cap = None  # OpenCV video capture object
        self.frame_counter = 0  # For frame skipping

        # Allowlist ViTForImageClassification for safe unpickling
        torch.serialization.add_safe_globals([ViTForImageClassification])

        # Load the PyTorch model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # put your model in torch.load()
        self.model = torch.load('your-model.pt', map_location=self.device, weights_only=False)
        self.model.eval()  # Set model to evaluation mode

        # Define emotion labels (ensure this matches your model's classes)
        self.emotions = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

        # Define preprocessing transformation
        self.transform = transforms.Compose([
            transforms.ToPILImage(),  # Convert OpenCV/numpy image to PIL
            transforms.Resize((224, 224)),  # Resize to match model input size
            transforms.ToTensor(),  # Convert to PyTorch tensor
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize RGB channels
        ])

        # start button
        self.btnStart.clicked.connect(self.start_video)
        # Stop button
        self.btnStop.clicked.connect(self.stop_video)
        # minimize window
        self.btnMinus.clicked.connect(self.showMinimized)
        # Close window
        self.btnClose.clicked.connect(self.close)

    def start_video(self):
        # Start webcam
        # I used droidcam for camera so i should put 1 in method but if you have camera on your system you have to use 0
        self.cap = cv2.VideoCapture(1)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)  # Reduce resolution
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(30)  # Update frame every 30ms

    def stop_video(self):
        self.timer.stop()
        if self.cap is not None:
            self.cap.release()
        self.lblCam.clear()

    def update_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            return

        self.frame_counter += 1
        if self.frame_counter % 3 != 0:  # Skip 2 out of every 3 frames
            return

        # Emotion recognition pipeline
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        batch_faces = []
        face_coords = []

        for (x, y, w, h) in faces:
            face = frame[y:y + h, x:x + w]  # Extract the face region
            rgb_face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
            batch_faces.append(self.transform(rgb_face))  # Preprocess face
            face_coords.append((x, y, w, h))  # Store face coordinates

        if batch_faces:
            batch_tensor = torch.stack(batch_faces).to(self.device)
            with torch.no_grad():
                preds = self.model(batch_tensor)  # Batch prediction

            # Safeguard against mismatch between faces and predictions
            if len(preds) != len(face_coords):
                print(f"Warning: Detected {len(face_coords)} faces but model returned {len(preds)} predictions.")

            # Process each face and prediction
            for i in range(min(len(preds), len(face_coords))):
                x, y, w, h = face_coords[i]
                try:
                    index = torch.argmax(preds[i]).item()  # Get predicted class index
                    if 0 <= index < len(self.emotions):  # Ensure valid index
                        emotion = self.emotions[index]
                    else:
                        emotion = "Unknown"  # Fallback for out-of-range indices

                    # Draw rectangle and emotion label
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                    cv2.putText(frame, emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
                except Exception as e:
                    print(f"Error processing prediction for face {i}: {e}")

        # Convert the frame to display in PyQt5
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_frame.shape
        bytes_per_line = ch * w
        qt_image = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
        self.lblCam.setPixmap(QPixmap.fromImage(qt_image))


    def closeEvent(self, event):
        self.stop_video()
        event.accept()

    def mousePressEvent(self, event):
        if event.button() == QtCore.Qt.LeftButton:
            self.offset = event.pos()
        else:
            super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self.offset is not None and event.buttons() == QtCore.Qt.LeftButton:
            self.move(self.pos() + event.pos() - self.offset)
        else:
            super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        self.offset = None
        super().mouseReleaseEvent(event)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = EmotionRecognition()
    window.show()
    sys.exit(app.exec_())
