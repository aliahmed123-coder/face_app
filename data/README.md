---
noteId: "ed6a3a7044d911f19ee061ef2b2307f7"
tags: []

---

# Dataset Sources and Preparation Guide

This document provides a comprehensive guide to face datasets for training and evaluating the Face Attendance System models.

---

## Face Recognition Datasets

### 1. LFW (Labeled Faces in the Wild)
- **Size**: 13,233 images of 5,749 people
- **Use**: Face verification benchmark
- **Download**: http://vis-www.cs.umass.edu/lfw/
- **Format**: JPEG, 250x250 pixels
- **License**: Research use

### 2. CelebA (CelebFaces Attributes)
- **Size**: 202,599 face images of 10,177 identities
- **Use**: Face detection, recognition, attribute prediction
- **Download**: https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html
- **Format**: JPEG, various sizes
- **License**: Non-commercial research

### 3. VGGFace2
- **Size**: 3.31 million images of 9,131 subjects
- **Use**: Large-scale face recognition training
- **Download**: https://www.robots.ox.ac.uk/~vgg/data/vgg_face2/
- **Format**: JPEG, various resolutions
- **License**: Creative Commons Attribution-ShareAlike 4.0

### 4. MS-Celeb-1M
- **Size**: 10 million images of 100,000 celebrities
- **Use**: Large-scale face recognition pre-training
- **Download**: https://www.microsoft.com/en-us/research/project/ms-celeb-1m-challenge-recognizing-one-million-celebrities-real-world/
- **Note**: Check current availability and license terms
- **License**: Research use (restricted)

### 5. CASIA-WebFace
- **Size**: 494,414 images of 10,575 subjects
- **Use**: Face recognition training
- **Download**: https://drive.google.com/open?id=1Of_EVz-yHV7QVWQGihYfvtny9Ne8G_-G
- **Format**: JPEG
- **License**: Research use

---

## Face Detection Datasets

### 6. WIDER FACE
- **Size**: 32,203 images with 393,703 labeled faces
- **Use**: Face detection (varying scales, occlusion, pose)
- **Download**: http://shuoyang1213.me/WIDERFACE/
- **Format**: JPEG with bounding box annotations
- **License**: Research use
- **Splits**: Train 40%, Val 10%, Test 50%

---

## Anti-Spoofing Datasets

### 7. NUAA Photograph Imposter Database
- **Size**: 12,614 images (5,105 live + 7,509 spoofed)
- **Use**: Print attack detection
- **Download**: Available via academic request
- **Attacks**: Printed photo attacks

### 8. CASIA-FASD (Face Anti-Spoofing Database)
- **Size**: 600 video clips of 50 subjects
- **Use**: Multi-modal anti-spoofing
- **Download**: http://www.cbsr.ia.ac.cn/english/FASDB_EN.asp
- **Attacks**: Warped photo, cut photo, video replay
- **Quality levels**: Low, normal, high

### 9. Replay-Attack Database
- **Size**: 1,300 video clips of 50 subjects
- **Use**: Video replay attack detection
- **Download**: https://www.idiap.ch/en/dataset/replayattack
- **Attacks**: Print, mobile phone display, tablet display
- **License**: Research use (requires agreement)

### 10. OULU-NPU
- **Size**: 4,950 videos from 55 subjects
- **Use**: Generalizable face presentation attack detection
- **Download**: https://sites.google.com/site/abordecentre/datasets
- **Attacks**: Print (2 printers), replay (2 displays)
- **Protocols**: 4 evaluation protocols
- **License**: Academic use

---

## Preprocessing Pipeline

### Recommended Steps

```
1. Face Detection & Alignment
   - Detect faces using MTCNN or RetinaFace
   - Extract 5 facial landmarks (eyes, nose, mouth corners)
   - Apply similarity transform to align faces

2. Resize
   - Standard sizes: 112x112 (ArcFace) or 160x160 (FaceNet)
   - Maintain aspect ratio with center crop

3. Augmentation (Training Only)
   - Horizontal flip (50% probability)
   - Random brightness/contrast (+/- 20%)
   - Random rotation (+/- 10 degrees)
   - Color jitter
   - Random erasing/cutout (simulates occlusion)
   - Motion blur (simulates camera movement)

4. Normalisation
   - Scale pixels to [0, 1] or [-1, 1]
   - Per-channel mean subtraction (if model requires)
```

### Quality Filtering
```
- Remove images with face size < 60x60 pixels
- Remove images with Laplacian variance < 50 (too blurry)
- Remove images with extreme pose (|yaw| > 60 degrees)
- Remove duplicates via perceptual hashing
```

---

## Directory Structure for Training

```
datasets/
├── recognition/
│   ├── train/
│   │   ├── person_001/
│   │   │   ├── 001.jpg
│   │   │   ├── 002.jpg
│   │   │   └── ...
│   │   ├── person_002/
│   │   └── ...
│   ├── val/
│   │   └── (same structure)
│   └── test/
│       └── pairs.txt          # verification pairs
│
├── detection/
│   ├── images/
│   │   ├── train/
│   │   └── val/
│   └── labels/                # YOLO format: class cx cy w h
│       ├── train/
│       └── val/
│
├── anti_spoofing/
│   ├── live/
│   │   ├── train/
│   │   └── val/
│   ├── print/
│   │   ├── train/
│   │   └── val/
│   ├── replay/
│   │   ├── train/
│   │   └── val/
│   └── mask/
│       ├── train/
│       └── val/
│
└── liveness/
    ├── blink_sequences/
    ├── head_turn_sequences/
    └── static_faces/
```

---

## YOLO Label Format

For face detection with YOLOv8, labels should be in YOLO format:

```
# Each line in a .txt file (one per image):
# class_id center_x center_y width height (normalised 0-1)
0 0.453 0.312 0.125 0.187
0 0.721 0.498 0.098 0.153
```

Where class 0 = face.

---

## Quick Start Commands

```bash
# Download LFW
wget http://vis-www.cs.umass.edu/lfw/lfw.tgz
tar -xzf lfw.tgz

# Download WIDER FACE
# Visit http://shuoyang1213.me/WIDERFACE/ and download manually

# Convert WIDER FACE to YOLO format (example script)
python scripts/wider_to_yolo.py --input wider_face/ --output datasets/detection/

# Train YOLOv8 face detector
yolo detect train data=face_dataset.yaml model=yolov8n.pt epochs=100 imgsz=640
```

---

## Notes

- Always verify dataset licenses before commercial use
- Some datasets require academic email or institutional agreement
- For production, consider combining multiple datasets for diversity
- Use stratified splits to ensure all identities appear in train/val/test
- Anti-spoofing models benefit from cross-dataset evaluation
