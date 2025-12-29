# Lung Disease Detection from Respiratory Sounds

<img width="775" height="575" alt="methdology flow" src="https://github.com/user-attachments/assets/903e00f5-cb39-425e-8daf-f004c1a5c463" />

## 📌 Overview
This repository implements a **hybrid deep learning framework** for automatic **multi-class lung disease detection** from respiratory sounds.  
The model integrates **deep audio features (mel-spectrogram + CNN–BiLSTM–Attention)** with **handcrafted acoustic features** (MFCCs, chroma, ZCR, spectral centroid, bandwidth).  
Explainability is achieved using **Grad-CAM, Integrated Gradients, and SHAP** for different feature branches.

### 🚀 Available Implementations
- **Python Version** (Original): Full-featured implementation with TensorFlow/Keras
- **JavaScript Version** (Node.js): TensorFlow.js implementation for server/edge deployment - see **[JAVASCRIPT_GUIDE.md](JAVASCRIPT_GUIDE.md)**

### Target Diseases
- Bronchial
- Asthma
- COPD
- Healthy
- Pneumonia

## Preprocessing Pipeline

<img width="961" height="553" alt="preprocessing pipeline" src="https://github.com/user-attachments/assets/db98f776-5449-4096-8060-15c25e596780" />


## 🏗️ Model Architecture

<img width="908" height="555" alt="model architecture" src="https://github.com/user-attachments/assets/c7537eca-e998-4d59-8f88-35b364203c5e" />


The model consists of two parallel branches:

1. **Mel-Spectrogram Branch**
   - Input: 4s audio → Mel-Spectrogram (128 × ~250)
   - 3 Conv2D blocks with BatchNorm, ReLU, MaxPooling, Dropout
   - Flattened via `TimeDistributed`
   - Bidirectional LSTM (128 units × 2 directions)
   - Additive Attention → temporal context vector
   - 256-dim embedding

2. **Handcrafted Feature Branch**
   - Features: MFCC, Chroma, ZCR, Spectral Centroid, Bandwidth
   - Total dimension ≈ 70
   - Fully connected network (Dense(256) → Dense(128))

3. **Fusion + Classification**
   - Concatenate embeddings (256 + 128 = 384)
   - Dense(256) + Dropout
   - Output Softmax layer (5 classes)


## ⚙️ Features
- End-to-end **deep + handcrafted feature fusion**
- Robust **data augmentations**: pitch shift, time-stretch, noise injection
- Explainable AI (XAI) methods:
  - **Grad-CAM** on mel spectrogram
  - **Integrated Gradients** on mel spectrogram
  - **SHAP** values on handcrafted features
- Evaluation metrics:
  - Accuracy, Loss, ROC-AUC, Confusion Matrix, Classification Report
  - Per-class AUC, Micro- and Macro-averaged ROC curves

## 📂 Dataset
- **Asthma Detection Dataset Version 2** (from Kaggle)
- Structure:
  ```
  dataset/
  ├── Bronchial/*.wav
  ├── asthma/*.wav
  ├── copd/*.wav
  ├── healthy/*.wav
  ├── pneumonia/*.wav
  ```

## 🚀 Training
- Optimizer: Adam (`lr=3e-4`, weight decay = 1e-4)
- Loss: Sparse Categorical Crossentropy
- Regularization: Dropout + Early Stopping
- Batch size: 16
- Epochs: 100 (with early stopping at 70)

## 📊 Results
- Strong validation and test accuracy across all classes
- ROC-AUC > 0.90 for all of the classes
- Grad-CAM & IG show meaningful attention on disease-relevant regions
- SHAP highlights important handcrafted features (MFCCs, spectral properties)

## 🔍 Explainability Examples
- **Grad-CAM** overlays class activation maps on mel-spectrograms
- **Integrated Gradients** highlights frequency bands most influential
- **SHAP** plots show feature importance of handcrafted features

## 📦 Installation

### Python Version
```bash
# Install dependencies (if on Colab/Kaggle, adjust as needed)
pip install numpy scipy pandas matplotlib seaborn librosa soundfile scikit-learn tensorflow==2.15.0 shap
```

### JavaScript Version (Node.js)
```bash
# Install Node.js dependencies
npm install
```

For detailed JavaScript installation and usage, see **[JAVASCRIPT_GUIDE.md](JAVASCRIPT_GUIDE.md)**

## ▶️ Usage

### Python Usage
1. Clone repo:
   ```bash
   git clone https://github.com/Pbrenya/Listening-to-the-Lungs.git
   cd Listening-to-the-Lungs
   ```
2. Prepare dataset under `data_dir` path inside `CFG` class
3. Run notebook or training script
4. Evaluate using built-in metrics
5. Visualize XAI results

### JavaScript Usage
```bash
# Run inference on audio file
node js/run.js infer --wav path/to/audio.wav

# Run training demo
npm run train

# Run XAI demo
npm run xai
```

See **[JAVASCRIPT_GUIDE.md](JAVASCRIPT_GUIDE.md)** for comprehensive documentation, examples, and case studies.

## 📈 Visualization
- **Training Curves**: Accuracy & loss over epochs
- **Confusion Matrix**: Per-class classification performance
- **ROC Curves**: One-vs-rest, micro/macro average
- **XAI Visualizations**: Grad-CAM overlays, Integrated Gradients, SHAP barplots

## 🧠 Future Work
- Expand dataset with more diseases (e.g., Tuberculosis, COVID-19 coughs)
- Deploy as **web app** with real-time inference
- Use **transformer-based encoders** (AST, Wav2Vec2) for stronger embeddings


## Citation

      @misc{saky2025explainablemultimodaldeeplearning,
            title={Explainable Multi-Modal Deep Learning for Automatic Detection of Lung Diseases from Respiratory Audio Signals}, 
            author={S M Asiful Islam Saky and Md Rashidul Islam and Md Saiful Arefin and Shahaba Alam},
            year={2025},
            eprint={2512.00563},
            archivePrefix={arXiv},
            primaryClass={cs.SD},
            url={https://arxiv.org/abs/2512.00563}, 
      }
