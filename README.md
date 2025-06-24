#  Speech Emotion Recognition Web App

This project is developed as part of the **MARS IIT Roorkee Open Projects 2025**. It focuses on recognizing human emotions from speech using machine learning and deep learning techniques.

---

## Description

The goal is to build an end-to-end pipeline that:
- Processes speech/audio data
- Extracts useful features
- Trains a machine learning model to classify emotions
- Deploys the model in a working Streamlit-based web app

The application predicts emotional states like **happy**, **sad**, **angry**, and **neutral** from `.wav` audio files uploaded by the user.

---

##  Dataset

We used the [RAVDESS dataset](https://zenodo.org/record/1188976), which includes high-quality audio clips of speech and song labeled with emotions. Only the following files were used for this project:

- `Audio_Speech_Actors_01-24`
- `Audio_Song_Actors_01-24`

The dataset contains `.wav` files labeled using encoded filenames which include emotion IDs.

---

##  Pre-processing

1. **File Parsing**: Audio files were loaded from nested Actor folders.
2. **Emotion Mapping**: The third number in the filename (e.g., `03-01-05-01-02-01-12.wav`) determines the emotion label.
3. **Audio Trimming**: Each clip was loaded using `librosa` with a fixed duration of 3 seconds.
4. **Feature Extraction**:
   - **MFCC (Mel Frequency Cepstral Coefficients)**
   - **Chroma Features**
   - **Mel Spectrogram**

These features were flattened and concatenated into a single feature vector.

---

##  Methodology

- **Language**: Python
- **Libraries Used**: `librosa`, `scikit-learn`, `joblib`, `streamlit`, `numpy`, `soundfile`
- **Steps**:
  1. Extract features from `.wav` files
  2. Split data into train and validation (80/20)
  3. Train a Random Forest Classifier
  4. Evaluate using F1-score and confusion matrix
  5. Save model to `emotion_model.pkl`
  6. Deploy model using Streamlit

---

##  Model Pipeline

```text
.wav file → Feature Extraction → Model → Emotion Prediction






