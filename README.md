# Module 3: Computer Vision Projects

A comprehensive collection of computer vision and deep learning laboratory exercises covering CNN visualization, object detection, and OCR processing.

## 📚 Project Overview

This module contains four distinct projects exploring different aspects of computer vision and AI:

### 1. **Lab 2.1: Understanding CNN Layers with VGG16**
**Location:** `Lab2_1_CNN_Visualization/`

Learn how Convolutional Neural Networks learn hierarchical representations through feature visualization.

- **Objective:** Extract and visualize feature maps from different layers of VGG16
- **Key Concepts:** CNN architecture, feature visualization, transfer learning, model interpretability
- **Technologies:** TensorFlow/Keras, VGG16, ImageNet pre-trained weights
- **Input:** Image files (JPG/PNG in `input_images/`)
- **Output:** Feature map visualizations (4×4 grids saved to `output_images/`)
- **Skills:** Model loading, layer extraction, visualization, explainable AI

**Getting Started:**
```bash
cd Lab2_1_CNN_Visualization
pip install -r requirements.txt
python cnn_visualization.py
```

---

### 2. **Lab 3.1: Pre-trained Object Detection with YOLO**
**Location:** `Lab3_1_YOLO_Object_Detection/`

Implement real-time multi-object detection using YOLOv8 for detecting all COCO dataset objects in videos.

- **Objective:** Build a production-ready object detection system
- **Key Concepts:** One-stage detectors, real-time processing, FPS optimization, video I/O
- **Technologies:** YOLOv8n (ultralytics), OpenCV, NumPy
- **Input:** Video files (MP4/AVI in `input_video/`) or webcam
- **Output:** Annotated video with detection bounding boxes, class labels, and statistics
- **Skills:** Pre-trained model usage, real-time inference, performance monitoring

**Getting Started:**
```bash
cd Lab3_1_YOLO_Object_Detection
pip install -r requirements.txt
python yolo_detection.py
```

---

### 3. **Lab 4.2: OCR Pipeline with Tesseract**
**Location:** `Lab4_2_OCR_Tesseract/`

Complete optical character recognition pipeline for extracting text and structured data from documents.

- **Objective:** Build end-to-end document processing system
- **Key Concepts:** Image preprocessing, OCR, structured data extraction, accuracy metrics
- **Technologies:** Tesseract OCR, OpenCV, Pillow, Regex, JSON
- **Input:** Document images (PNG/JPG in `input_documents/`)
- **Output:** Extracted text in JSON format with accuracy reports
- **Skills:** Image preprocessing, text recognition, data extraction, automation

**Getting Started:**
```bash
cd Lab4_2_OCR_Tesseract
pip install -r requirements.txt
python ocr_pipeline.py
```

---

### 4. **Lab 2.1 Interactive Notebook**
**Location:** `Lab2_1_Student_Interactive.ipynb`

Interactive Jupyter notebook for exploring CNN visualization concepts step-by-step.

- **Format:** Jupyter Notebook (.ipynb)
- **Content:** Interactive cells with explanations and code
- **Use:** Learning and exploration

---

## 🗂️ Workspace Structure

```
Module3_Labs/
├── README.md                                    # This file
├── .gitignore                                   # Git ignore for virtual environments
│
├── Lab2_1_Student_Interactive.ipynb             # Interactive notebook
│
├── Lab2_1_CNN_Visualization/
│   ├── cnn_visualization.py                     # Main script
│   ├── requirements.txt
│   ├── README.md
│   ├── answers.txt
│   ├── input_images/
│   └── output_images/
│
├── Lab3_1_YOLO_Object_Detection/
│   ├── yolo_detection.py                        # Main script
│   ├── requirements.txt
│   ├── README.md
│   ├── answers.txt
│   ├── yolov8n.pt                              # Pre-trained model
│   ├── input_video/
│   ├── output_video/
│   └── output_frames/
│
└── Lab4_2_OCR_Tesseract/
    ├── ocr_pipeline.py                          # Main script
    ├── requirements.txt
    ├── README.md
    ├── answers.txt
    ├── SETUP.txt                               # Tesseract setup guide
    ├── accuracy_report.txt
    ├── input_documents/
    ├── output_images/
    └── output_json/
```

---

## 🔧 Common Setup Steps

### 1. System Requirements
- **Python:** 3.8 or higher
- **OS:** Windows, macOS, or Linux
- **RAM:** 4GB minimum
- **GPU:** Optional (for faster processing)

### 2. Create Virtual Environment
```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

### 3. Install Project Dependencies
Each project has its own `requirements.txt`. Install dependencies for the project you want to run:

```bash
cd <ProjectFolder>
pip install -r requirements.txt
```

### 4. Special Setup (Lab 4.2 OCR)
Lab 4.2 requires Tesseract OCR system installation:
- See `Lab4_2_OCR_Tesseract/SETUP.txt` for detailed installation instructions
- Windows: Download installer from [Tesseract releases](https://github.com/UB-Mannheim/tesseract/wiki)
- macOS: `brew install tesseract`
- Linux: `sudo apt-get install tesseract-ocr`

---

## 🚀 Running the Projects

### Lab 2.1 - CNN Visualization
```bash
cd Lab2_1_CNN_Visualization
python cnn_visualization.py
# Outputs visualizations to output_images/
```

### Lab 3.1 - YOLO Detection
```bash
cd Lab3_1_YOLO_Object_Detection
python yolo_detection.py
# Processes input_video/test.mp4
# Outputs annotated video to output_video/
```

### Lab 4.2 - OCR Pipeline
```bash
cd Lab4_2_OCR_Tesseract
python ocr_pipeline.py
# Processes documents from input_documents/
# Outputs JSON files to output_json/
```

### Lab 2.1 - Interactive Notebook
```bash
jupyter notebook Lab2_1_Student_Interactive.ipynb
```

---

## 📊 Key Technologies

| Project | Framework | Models | Input | Output |
|---------|-----------|--------|-------|--------|
| Lab 2.1 CNN | TensorFlow/Keras | VGG16 | Images | Feature visualizations |
| Lab 3.1 YOLO | Ultralytics | YOLOv8n | Video/Webcam | Annotated video |
| Lab 4.2 OCR | Tesseract | N/A | Document images | JSON + text |

---

## 🎯 Learning Objectives

After completing all projects, students will understand:

✅ **Deep Learning Fundamentals**
- CNN architecture and layer operations
- Feature hierarchies in neural networks
- Transfer learning and pre-trained models

✅ **Computer Vision Applications**
- Real-time object detection
- One-stage vs two-stage detectors
- Video processing and frame analysis

✅ **Production Code**
- Handling edge cases and errors
- Performance optimization
- Result reporting and validation

✅ **Practical Skills**
- Using pre-trained models effectively
- Processing various media types (images, videos, documents)
- Extracting structured data from unstructured content

---

## 📝 Next Steps

1. **Start with Lab 2.1** - Understand CNN fundamentals
2. **Move to Lab 3.1** - Apply knowledge to real-time detection
3. **Complete Lab 4.2** - End-to-end practical system
4. **Explore variations** - Modify models, parameters, and inputs

---

## 📖 Additional Resources

- [TensorFlow Documentation](https://www.tensorflow.org/)
- [YOLOv8 Docs](https://docs.ultralytics.com/)
- [OpenCV Tutorials](https://docs.opencv.org/)
- [Tesseract OCR](https://github.com/UB-Mannheim/tesseract/wiki)

---

## 💡 Tips

- Always activate your virtual environment before running projects
- Check individual project README.md files for detailed documentation
- Review answers.txt files to understand expected outputs
- Start with sample data provided in input folders

---

**Happy Learning! 🚀**
