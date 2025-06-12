# HumanDetectorApp: Real-Time Human Detection with YOLO

## Overview

The `HumanDetectorApp` is a Python application built with KivyMD and OpenCV that performs real-time human detection using the YOLO (You Only Look Once) model. It supports live RTSP streams, video files, and static images, providing a user-friendly interface to visualize detected objects with bounding boxes and labels. The app is designed for applications like surveillance, search and rescue, or monitoring, with robust error handling for RTSP-related issues such as frame delay, distortion, and H.264 decoding errors.

## Features

- **Live Detection**: Processes RTSP streams or webcam feeds in real-time (`LiveTab`).
- **Video Playback**: Analyzes video files with seek functionality (`VideoTab`).
- **Image Processing**: Detects objects in static images (`ImageTab`).
- **YOLO Integration**: Uses a pre-trained YOLO model (`best.onnx`) for efficient human detection.
- **Error Handling**: Mitigates RTSP issues (e.g., H.264 errors like `corrupted macroblock`) with TCP transport, frame validation, and stream reinitialization.
- **Optimized Performance**: Reduces frame delay by running YOLO every 4th frame in live mode and using larger queues (`frame_queue=10`, `result_queue=20`).
- **User Interface**: KivyMD-based GUI with tabs, buttons, sliders, and error labels for intuitive control.

## How It Works

### Architecture

The app is structured around a KivyMD interface with three tabs:

1. **LiveTab**:
   - Captures frames from a webcam or RTSP stream using OpenCV’s `cv2.VideoCapture`.
   - Processes frames with YOLO for human detection every 4th frame to reduce latency.
   - Displays annotated frames with bounding boxes and labels.
   - Handles RTSP issues with TCP transport, increased buffering (`CAP_PROP_BUFFERSIZE=10`), and reinitialization after 3 decode errors.

2. **VideoTab**:
   - Loads video files (e.g., `.mp4`, `.avi`) and supports play/pause and seeking via a slider.
   - Runs YOLO on every frame with a confidence threshold of 0.65.
   - Updates a time label to show current/total duration.
   - Uses similar error handling for video decoding issues.

3. **ImageTab**:
   - Processes single images (e.g., `.jpg`, `.png`) with YOLO (confidence threshold 0.76).
   - Saves the annotated output to `output.jpg`.
   - Displays results without real-time constraints.

### Feature Extraction

Feature extraction is performed by the YOLO model’s convolutional neural network (CNN) backbone, which transforms raw frames (640x480) into feature maps encoding edges, shapes, and semantic patterns. The process involves:

- **Input**: BGR frames from OpenCV.
- **Backbone**: CSPDarknet or similar extracts multi-scale feature maps.
- **Feature Pyramid Network (FPN)**: Combines features for detecting objects of varying sizes.
- **Output**: Bounding boxes, class IDs (e.g., human), and confidence scores.
- **Usage**: The `draw_results` method visualizes predictions by drawing colored bounding boxes (e.g., green for humans) and labels on frames.

Feature extraction is optimized for live RTSP by skipping YOLO on 3 out of 4 frames, reducing computational load and frame delay. Invalid frames (e.g., from H.264 errors) are replaced with a fallback black frame to prevent distortion.

### RTSP Issue Mitigation

The app addresses RTSP-related challenges:

- **Frame Delay**: Reduced by:
  - Running YOLO every 4th frame (`frame_count % 4 == 0`).
  - Using larger queues (`frame_queue=10`, `result_queue=20`) to buffer frames.
  - Setting `CAP_PROP_FPS=30` to align with processing capabilities.
- **Frame Distortion**: Mitigated by:
  - Validating frame shape and content (`frame.shape == (480, 640, 3) and np.any(frame)`).
  - Using a fallback frame during errors.
- **H.264 Errors** (e.g., `corrupted macroblock 115 27`):
  - Forces TCP transport (`rtsp_transport;tcp`) for reliable packet delivery.
  - Increases buffer size (`CAP_PROP_BUFFERSIZE=10`).
  - Reinitializes the stream after 3 consecutive decode errors.
  - Retries frame reads (3 attempts) to handle temporary packet loss.

### Threading and Queues

- **Threading**: A separate `processing_thread` (`process_live` or `process_video`) handles frame capture and YOLO inference, while the main thread updates the GUI via `update_live` or `update_video`.
- **Queues**:
  - `frame_queue` (maxsize=10): Buffers raw frames to prevent dropping.
  - `result_queue` (maxsize=20): Stores processed frames for display.
  - Conservative clearing (keeps 1-2 frames) preserves I-frames, reducing H.264 errors.

### Visualization

Processed frames are converted to RGB, flipped vertically, and rendered as Kivy textures in `update_texture`. The GUI updates at 30 FPS for live feeds, ensuring smooth display despite YOLO’s selective processing.

## Requirements

- **Python**: 3.8 or higher
- **Dependencies** (install via `pip`):
  ```bash
  pip install kivymd opencv-python ultralytics numpy plyer
  ```
- **YOLO Model**: Pre-trained `best.onnx` file (place at `HumanDetection\UI\best.onnx` or update `model_path` in code).
- **FFmpeg**: Required for RTSP support. Ensure it’s installed and accessible:
  - Windows: Add FFmpeg to PATH (download from [ffmpeg.org](https://ffmpeg.org)).
  - Linux/Mac: Install via package manager (e.g., `sudo apt-get install ffmpeg`).
- **Hardware**: CPU with at least 4 cores recommended; GPU (CUDA) optional for faster YOLO inference.
- **Network**: Stable connection (>1 Mbps) for RTSP streams.

## Setup Instructions

1. **Clone or Download the Project**:
   - Save the project files, including `humanDetector.py`, to a local directory (e.g., `C:\Projects\HumanDetectorApp`).

2. **Install Dependencies**:
   ```bash
   pip install kivymd opencv-python ultralytics numpy plyer
   ```

3. **Place YOLO Model**:
   - Ensure `best.onnx` is at `HumanDetection\UI\best.onnx`.
   - Alternatively, modify `model_path` in `human_detector_app.py`:
     ```python
     model_path = 'path/to/your/best.onnx'
     ```

4. **Install FFmpeg**:
   - **Windows**: Download FFmpeg, extract, and add the `bin` folder to your system PATH.
   - **Linux**: `sudo apt-get install ffmpeg`
   - **Mac**: `brew install ffmpeg`

5. **Verify OpenCV FFmpeg Support**:
   ```python
   import cv2
   print(cv2.getBuildInformation())
   ```
   - Look for `FFMPEG: YES` in the output.

## Running the Application

1. **Navigate to Project Directory**:
   ```bash
   cd C:\Projects\HumanDetectorApp
   ```

2. **Run the Script**:
   ```bash
   python human_detector_app.py
   ```

3. **Using the Interface**:
   - **LiveTab**:
     - Enter an RTSP URL (e.g., `rtsp://wowzaec2demo.streamlock.net/vod/mp4:BigBuckBunny_115k.mov`) or click "Camera" for webcam.
     - Click "RTSP" or "Camera" to start; "Stop" to halt.
   - **VideoTab**:
     - Click "Load Video" to select a `.mp4`, `.avi`, or `.mov` file.
     - Use "Play/Pause" and the slider to control playback.
   - **ImageTab**:
     - Click "Load Image" to select a `.jpg`, `.jpeg`, or `.png` file.
     - View results; annotated image saves as `output.jpg`.

## Testing and Troubleshooting

### Testing
- **RTSP Stream**:
  - Verify stream stability in VLC:
    ```bash
    vlc rtsp://your_rtsp_url
    ```
  - Monitor logs for H.264 errors or retries (check console or log file).
- **Video Playback**:
  - Test seek functionality and time label updates.
- **Image Processing**:
  - Confirm `output.jpg` contains bounding boxes.
- **Latency**:
  - Compare real-time events (e.g., webcam motion) with display to measure delay.

### Common Issues
- **H.264 Errors** (e.g., `corrupted macroblock`):
  - Ensure TCP transport is enabled (check `os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"]`).
  - Test with a stable RTSP stream.
  - Update OpenCV: `pip install --upgrade opencv-python opencv-contrib-python`.
- **Frame Delay**:
  - Use a lighter YOLO model (e.g., `yolov8n.onnx`).
  - Enable GPU: Modify `self.model = YOLO(model_path, device='cuda')`.
- **FFmpeg Not Found**:
  - Verify FFmpeg is in PATH and OpenCV supports it.
- **Model Path Error**:
  - Update `model_path` to match your `best.onnx` location.

## Limitations
- **Performance**: CPU-based YOLO inference may cause slight delays on low-end hardware. GPU acceleration is recommended.
- **RTSP Stability**: Dependent on network quality; unstable streams may trigger reinitialization.
- **Memory Usage**: Larger queues increase memory (~20 MB for 20 frames at 640x480).

## Future Improvements
- **Lighter Model**: Use YOLOv8n for faster inference.
- **Adaptive YOLO**: Dynamically adjust YOLO frequency based on CPU load.
- **Frame Preprocessing**: Resize frames to 320x240 before YOLO to reduce computation.
- **Error Logging**: Save logs to a file for easier debugging.

## License
This project is for educational purposes. Ensure compliance with licenses for dependencies (KivyMD, OpenCV, ultralytics) and the YOLO model.

## Contact
For issues or contributions, contact the project maintainer or open an issue on the repository (if hosted).