# HandGestureRecognition using 3DConv CNN RNN

## Abstract
This project focuses on developing a gesture recognition system for smart TVs, enabling users to control functionalities like volume adjustment, video navigation, and playback control using hand gestures. The system recognizes five specific gestures—thumbs up, thumbs down, left swipe, right swipe, and stop—each mapped to a TV command. Various deep learning architectures, including Conv3D, Conv2D + GRU, Conv2D + LSTM, and MobileNet + LSTM (transfer learning), were evaluated to determine the best-performing model.

## Problem Statement
As user interaction technology advances, gesture recognition offers a hands-free, intuitive alternative to remote controls. The goal is to design a model that can accurately classify gestures captured by a webcam, allowing for seamless smart TV control. The challenge lies in developing a model that is both accurate and computationally efficient, suitable for real-time applications.

## Data Source
The dataset consists of short video clips, each representing one of the five gestures. Each video is divided into 30 frames, simulating the real-world application of gesture recognition through continuous frame capture. The videos were recorded under various lighting conditions and include different individuals performing the gestures, ensuring diversity.

*Dataset:* https://drive.google.com/uc?id=1ehyrYBQ5rbQQe6yL4XbLWe3FMvuVUGiL

### Sample image from Dataset
![sample image](https://github.com/Pavani89/HandGestureRecognition/blob/main/sample_image_from_dataset.png)

## Architecture Overview
1. **Conv3D**: This model uses 3D convolution to process temporal information in video data. While effective in capturing spatial-temporal features, the high parameter count led to overfitting and moderate performance.
2. **Conv2D + GRU**: Combines Conv2D for spatial feature extraction with a GRU layer for temporal sequence processing. This model reduced parameters but struggled to generalize effectively.
3. **Conv2D + LSTM**: Utilizes Conv2D for spatial features, followed by an LSTM for temporal recognition, achieving better accuracy and generalization than Conv3D and Conv2D + GRU.
4. **Transfer Learning: MobileNet + LSTM**: MobileNet extracts spatial features, and an LSTM captures temporal dependencies, achieving the highest accuracy with a minimal trainable parameter count due to transfer learning.

## Steps

### 1. **Data Preprocessing**
   - Resizing, cropping, and normalizing images to enhance gesture distinction and reduce background noise.
   - Data augmentation (slight rotations) to improve model robustness and account for variation in hand positioning.

### 2. **Model Training and Evaluation**
   - **Conv3D**: Baseline model with high computational cost and moderate accuracy (validation accuracy: 46%).
   - **Conv2D + GRU**: Improved parameter efficiency but low generalization (validation accuracy: 25%).
   - **Conv2D + LSTM**: Better accuracy and reduced overfitting (validation accuracy: 69%).
   - **MobileNet + LSTM**: Best model with high validation accuracy (74%) and minimal overfitting, achieved through transfer learning.

### 3. **Optimization Techniques**
   - Used Adam optimizer, batch normalization, dropout layers, and early stopping to enhance convergence and minimize overfitting.
   - Adjusted batch sizes based on GPU constraints for optimal computational efficiency.

### 4. **Evaluation**
   - Model performance was assessed based on training/validation accuracy, validation loss, and parameter efficiency.
   - MobileNet + LSTM achieved the best balance of accuracy and computational efficiency, with 86.1% training accuracy, 74% validation accuracy, and a validation loss of 0.60.

## Results and Conclusion
The MobileNet + LSTM model was selected as the final model, demonstrating strong accuracy and low parameter count, ideal for real-time application. Future improvements may explore advanced architectures and hyperparameter tuning for enhanced performance.
