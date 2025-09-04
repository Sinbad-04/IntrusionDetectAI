


**VisionCount** is a Computer Vision project for detecting and counting objects in video streams.  
The tool processes an input video and produces an output video with bounding boxes and object counts.

---


![Watch the video](https://raw.githubusercontent.com/Sinbad-04/VisionCount/main/output_video.mp4)


## ✨ Features

- 🎥 Detect and count objects directly from video (`.mp4`, `.avi`, etc.)
- 🔄 Export processed video with bounding boxes and counts
- 🧠 Pre-trained model available (`model.pt`)
- ⚡ Supports both real-time processing and offline video files
- 🧪 Separate test/demo script for quick validation

---

## 🛠️ Requirements

- Python **3.8+**
- Required libraries:
  - `torch`
  - `opencv-python`
  - `numpy`
  - `matplotlib` *(optional, for visualization)*
  - `tqdm` *(for progress bar)*
 
  ---

  🚀 Installation & Quick Start

  
1️⃣ Clone the repository
```bash
git clone https://github.com/Sinbad-04/VisionCount.git
cd VisionCount
```

2️⃣ Install dependencies
```bash
pip install -r requirements.txt
```
3️⃣ Run the main script

Process a video and export the result:
```bash
python main.py --input input_video.mp4 --output output_video.mp4
```
4️⃣ Run the test/demo script
```bash
python test.py
```

📂 Project Structure
VisionCount/
├── main.py              # Main script: video processing & object counting

├── test.py              # Test/demo script

├── model.pt             # Pre-trained model

├── output_video.mp4     # Sample output video

├── output_video.avi     # Alternative sample output

├── requirements.txt     # Dependency list

└── README.md            # This documentation


---
📝 Usage Examples

Count objects in a video:
```bash

python main.py --input street.mp4 --output result.mp4
```

Quick demo run:
```bash
python test.py
```

The processed video will be saved in the project folder with bounding boxes and object counts overlayed.







