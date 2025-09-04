# IntrusionDetectAI

 **IntrusionDetectAI** is a system that uses **YOLOv8** to detect intrusions or suspicious objects in video streams.  
It can process input videos, run object detection, and generate annotated output videos.

## 🚀 Features
- Real-time intrusion/object detection using `yolov8s.pt`
- Processes video input and outputs annotated results
- Includes sample videos (`vd1.mp4`) for testing

⚡ Installation & Usage

Clone the repository:
```bash
git clone https://github.com/Sinbad-04/IntrusionDetectAI.git
cd IntrusionDetectAI
```

Install dependencies:
```bash
pip install -r requirements.txt
```

Run detection on a sample video:
```bash
python main.py --input vd.mp4 --output result.mp4
```

(Adjust the arguments if different.)

📂 Project Structure

main.py — main script for video processing and YOLOv8 inference

yolov8s.pt — pre-trained YOLOv8 model

vd.mp4, vd1.mp4 — sample input videos

result.mp4 — sample output video with detections

🎯 Input / Output

Input: Video file (e.g., .mp4, .avi) containing the monitored area

Output: Annotated video with bounding boxes / intrusion alerts

🤝 Contributing

Contributions are welcome!

Fork the repository and create a new branch

Submit a pull request with a clear description

Make sure to test before submitting


