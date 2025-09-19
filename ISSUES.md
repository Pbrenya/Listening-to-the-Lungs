# Suggested Issues to open on GitHub

1) Add model export and inference script
Title: "Add model export and inference script"
Description: Implement `src/export.py` or `examples/inference_example.py` that loads trained model and provides `predict(wav_path)` plus optional Grad-CAM output.

2) Add unit tests for preprocessing
Title: "Unit tests for padding and feature extraction"
Description: Add tests verifying `pad_or_trim`, `compute_melspec`, `compute_handcrafted` produce expected shapes.

3) Implement TFRecord preprocessing
Title: "Precompute mel spectrograms into TFRecord"
Description: Add script to precompute features to speed up training.

4) Add TensorBoard logging
Title: "Add TensorBoard callback and log directory"
Description: Integrate TensorBoard logging in training to track metrics and Grad-CAM images.

5) Add deployment demo (FastAPI)
Title: "Create FastAPI demo for inference"
Description: Minimal web app to upload WAV and return prediction + Grad-CAM image.

6) Cross-validation experiments
Title: "Add k-fold cross-validation experiment"
Description: Implement Stratified K-Fold training script and aggregate metrics.

7) Add SHAP result caching & visualization improvements
Title: "Improve SHAP plotting: beeswarm, caching"
Description: Save SHAP values and add nicer plots for paper.
