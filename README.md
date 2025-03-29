# StrikeHer - AI-powered motion tracking system for self-defense training

Traditional self-defense training methods are often inaccessible, leaving many women without essential skills to protect themselves. StrikeHer is a real-time AI-powered self-defense training assistant. It utilizes **MediaPipe** for motion tracking and provides instant feedback on techniques, helping users improve their strikes, stances, and overall form. We aim to provide a sensor-free, interactive experience to improve accessibility and ease of use.

## 🚀 Features  
- **Hand & Pose Tracking**: Uses MediaPipe to analyze body movements in real time.  
- **Strike Detection**: Identifies key self-defense moves like punches, blocks, and jabs.  
- **Live Feedback**: The system analyzes form and provides real-time corrections using color and number-coded feedback for better posture. "90% correct stance → raise your arm higher!"
- **Camera Integration**: AI detects body posture and actions using a webcam or phone camera.

## 🛠️ Tech Stack  
- **Python**  
- **OpenCV**  
- **MediaPipe** (for pose estimation & hand tracking)  

## 📂 Project Structure  
| File | Description |
|------|------------|
| `feedback.py` | Provides AI-generated feedback based on movement analysis. |
| `overlay_mediapipe.py` | Displays tracking overlays on camera feed. |
| `test_camera*.py` | Tests camera input for real-time motion tracking. |
| `test_fist.py`, `test_jab.py` | Detects specific self-defense movements. |
