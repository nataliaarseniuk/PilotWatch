# PilotWatch - Pilot Authentication & Monitoring System

Real-time pilot presence detection system using computer vision and machine learning for drone safety monitoring.

## Overview

PilotWatch is a safety monitoring system that uses a convolutional neural network (CNN) to detect whether a pilot is present at a drone control station. The system provides real-time alerts if the operator leaves during an active flight session.

## Features

- 🎯 Real-time pilot presence detection (30 FPS)
- 🧠 Custom-trained CNN model (58% validation accuracy)
- 📹 Live camera feed with visual indicators
- ⚠️ Automated alert system for unattended stations
- 🔌 Hardware integration via CRSF serial protocol
- 📊 Telemetry display and flight session monitoring

## System Architecture

Camera → CNN Model → Pilot Detection → Alert System → GUI Display

## Tech Stack

- **Computer Vision**: OpenCV (cv2)
- **Machine Learning**: TensorFlow/Keras
- **GUI**: Tkinter
- **Hardware**: PySerial (CRSF protocol)
- **Language**: Python 3.10+

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/PilotWatch.git
cd PilotWatch
```

2. Create virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### 1. Collect Training Data

```bash
python src/collect_presence_data.py
```

- Press 'p' to capture pilot present images
- Press 'n' to capture no pilot images
- Press 'q' to quit

### 2. Train the Model

```bash
python src/train_presence_model.py
```

This will:
- Load images from `pilot_presence_dataset/`
- Train CNN for 20 epochs
- Save best model as `pilot_presence_model.h5`
- Generate training graphs

### 3. Run the Application

```bash
python src/drone_control_app.py
```

**Controls:**
- Click "Start Flight Session" to begin monitoring
- System alerts if pilot leaves during active session
- Click "Clear Alerts" to dismiss warnings

## Model Architecture

- **Input**: 128x128x3 RGB images
- **Architecture**: 3 Conv blocks (32→64→128 filters) + 2 Dense layers
- **Output**: Binary classification (pilot present/absent)
- **Training Data**: 171 images (71 pilot, 100 no-pilot)
- **Performance**: 88% validation accuracy

## Project Structure

PilotWatch/
├── src/
│   ├── collect_presence_data.py    # Data collection script
│   ├── train_presence_model.py     # Model training
│   └── drone_control_app.py        # Main application
├── pilot_presence_dataset/         # Demo images
├── requirements.txt                # Python dependencies
└── README.md

## Hardware Integration

The system integrates with RadioMaster Pocket via CRSF protocol over USB serial. While the serial connection works, RF transmission operates independently and would require firmware configuration or hardware interlock for physical drone control.

## Challenges & Learnings

This project went through multiple iterations:

1. **Gesture Recognition** (Failed) - 33% accuracy, gestures too similar
2. **Face Authentication** (Overfitted) - 99% validation, 0% real-world
3. **Presence Detection** (Success) - Simple binary classification that works

**Key Lessons:**
- Data quality > quantity
- Validation metrics ≠ real-world performance
- Iterative development is essential in ML

## Future Improvements

- [ ] Larger, more diverse dataset (1000+ images)
- [ ] Multi-pilot authentication (recognize specific individuals)
- [ ] Flight controller integration for automatic disarming
- [ ] Mobile app version
- [ ] Cloud logging and analytics

## Demo

https://www.youtube.com/watch?v=V858oQNAw4g

## License

MIT License - feel free to use for educational purposes

## Author

Natalia Arseniuk - [LinkedIn](www.linkedin.com/in/natalia-arseniuk/)

## Acknowledgments

Built as part of COMP4949 Assignment 2 - Big Data Analytics

---

**Questions or suggestions? Feel free to open an issue!**
