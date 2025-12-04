# deepfake_detector
A deepfake image detection model built using ResNet18 and transfer learning, capable of classifying real vs. AI-generated images with a lightweight and efficient pipeline.


Project Overview

This project implements a deepfake image classification model using transfer learning with ResNet-18.
The goal is to detect whether a given image is Real or Fake, based on frame-level analysis.

The system loads image data, trains a binary classifier, and evaluates it using accuracy, classification reports, and confusion matrices.

---

Dataset

The dataset used for this project was sourced from:

Kaggle — FaceForensics Dataset:
[https://www.kaggle.com/datasets/hungle3401/faceforensics](https://www.kaggle.com/datasets/hungle3401/faceforensics)

The dataset contains folders of:

 `real_data/` — authentic face images
 `fake_data/` — manipulated or altered face images

These folders are organized under a main directory named:

```
Dataset/
    real_data/
    fake_data/
```

The code automatically loads data for:

Training
Validation
Testing

Each split is expected to follow the same folder structure.

---

Model

The model is based on:

ResNet-18 (pretrained on ImageNet)
 Final fully-connected layer modified for binary classification (Real vs Fake)

This approach provides strong performance even with limited training data.

---

Features

This project includes several advanced components:

 Custom PyTorch `Dataset` class
 Automatic train/validation/test splitting
 Transfer learning with ResNet-18
 GPU acceleration when available
 Data augmentation during training
 Accuracy, loss tracking per epoch
 Final performance evaluation
 Classification report
 Confusion matrix

---

How to Run the Code

1. Install Dependencies

```bash
pip install torch torchvision scikit-learn pillow numpy
```

2. Organize Your Dataset

Place the dataset in a folder named `Dataset/` or specify a custom path using `--data_root`.

Expected structure:

```
Dataset/
    real_data/
        image1.jpg
        image2.jpg
        ...
    fake_data/
        image1.jpg
        image2.jpg
        ...
```

3. Run the Training Script

```bash
python main.py --epochs 10 --batch_size 16 --lr 1e-4
```

Optional arguments:

 `--data_root`  (default: Dataset)
 `--epochs`
 `--batch_size`
 `--lr` (learning rate)

---

Results

The final model performance (including classification report and confusion matrix) is shown in the "results" image included in the zip file submitted with this project.

This image reflects the test accuracy, precision/recall scores, and the confusion matrix produced after training.

---

File Structure

```
main.py                     # Main script
DeepfakeDataset class       # Custom dataset loader
DeepfakeDetector class      # ResNet18-based classifier
train_one_epoch()           # Training loop
evaluate()                  # Validation/test evaluation
results/                    # Contains final result screenshots (submitted separately)
```

---

Potential Improvements

Future work could include:

Multi-frame sequence analysis (video-level classification)
Additional augmentations
More advanced CNN or transformer-based models
Saving/loading model checkpoints
Grad-CAM visualizations for interpretability

---

License

This project uses publicly available datasets and libraries under their respective licenses.
The Kaggle dataset is subject to Kaggle’s usage rules.





