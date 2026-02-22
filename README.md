# YOLOSoccer: Vision-Based Analytics for Soccer Videos

An advanced computer vision and machine learning pipeline for analyzing soccer match footage. Built with Python, YOLO (Ultralytics), and OpenCV, this project automatically tracks players, referees, and the ball while extracting deep analytics such as team ball possession, player speed, and total distance covered.

It includes both a standalone processing script and an interactive web application built with Streamlit.

## 🚀 Features

* **Multi-Object Tracking:** Detects and tracks players, referees, and the soccer ball across frames using YOLO and Supervision's ByteTrack implementation.
* **Team Assignment:** Automatically separates players into their respective teams based on jersey colors using clustering.
* **Ball Possession Analytics:** Calculates which player has the ball and dynamically updates the overall team ball control percentages.
* **Camera Movement Estimation:** Compensates for camera panning and zooming to ensure accurate player tracking and real-world measurements.
* **Perspective Transformation:** Transforms camera-view coordinates to a top-down 2D pitch view for accurate spatial analysis.
* **Speed & Distance Estimation:** Calculates the real-time speed of players and the total distance they have covered during the clip.
* **Interactive Web App:** Includes a Streamlit interface for uploading custom videos or running the pipeline on built-in demo videos.

## 🛠️ Tech Stack

* **Language:** Python
* **Computer Vision:** OpenCV, Ultralytics (YOLO)
* **Tracking & Annotations:** Supervision
* **Data Manipulation:** NumPy, Pandas
* **Web UI:** Streamlit
* **Video Processing:** FFmpeg

## ⚙️ Prerequisites

* Python 3.8+
* [FFmpeg](https://ffmpeg.org/download.html) must be installed on your system and added to your system's PATH (used for video compression).

## 📦 Installation

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/sreesh2411/YOLOSoccer-Vision-based-analytics-for-soccer-videos.git](https://github.com/sreesh2411/YOLOSoccer-Vision-based-analytics-for-soccer-videos.git)
   cd YOLOSoccer-Vision-based-analytics-for-soccer-videos
   ```

2. **Install the required Python packages:**
   *(Note: Ensure you have your virtual environment activated if you use one)*
   ```bash
   pip install ultralytics supervision opencv-python numpy pandas streamlit
   ```

3. **Download YOLO Weights:**
   Ensure your YOLO weights file (`best.pt`) is placed in the `models/` directory.

## 💻 Usage

### 1. Running the Streamlit Web App (Recommended)
The easiest way to interact with the pipeline is through the Streamlit web interface, which allows you to upload MP4 videos and view the processed outputs directly in your browser.

```bash
streamlit run app.py
```

### 2. Running the Standalone Script
If you want to run the pipeline directly from the command line, you can use `main.py`. Ensure your input video is located at `input_videos/0bfacc_3.mp4` (or update the path in the script).

```bash
python main.py
```
* **Output:** The script generates an annotated video and a compressed version in the `output_videos/` directory. Stub files are saved in the `stubs/` directory to speed up subsequent runs on the same video.

## 📂 Project Structure

* `main.py`: The main execution script for processing videos via the command line.
* `app.py`: The Streamlit web application.
* `trackers/`: Contains the `Tracker` class handling YOLO detections, ByteTrack integration, and bounding box interpolations.
* `camera_movement_estimator/`: Logic for estimating and adjusting for camera movement.
* `player_ball_assigner/`: Determines which player is currently in possession of the ball.
* `speed_and_distance_estimator/`: Calculates player speed and distance covered over time.
* `team_assigner/`: Logic for assigning players to teams based on visual features.
* `view_transformer/`: Handles perspective transforms for top-down positional accuracy.
* `utils/`: Helper functions for video reading/writing and bounding box math.

## 🤝 Contributing
Contributions, issues, and feature requests are welcome! Feel free to check the issues page.
