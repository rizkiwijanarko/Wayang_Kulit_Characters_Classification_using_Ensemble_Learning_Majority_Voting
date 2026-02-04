# Wayang Classifier Ensemble

A deep learning project for classifying Indonesian Wayang characters using an ensemble of convolutional neural networks (CNNs). This repository includes the training pipeline and a web-based application for real-time classification.

## Overview
This project leverages Ensemble Learning with three powerful architectures:
- **InceptionV3**
- **EfficientNetB2**
- **ResNet101V2**

By combining the predictions of these three models, the system achieves more robust and accurate classification of various Wayang characters compared to a single model approach.

## Features
- **Ensemble Inference**: Uses majority voting from three different pre-trained models.
- **Web Interface**: Simple Flask-based web application to upload images and get instant classification.
- **Training Pipeline**: Comprehensive Jupyter Notebook covering data preprocessing, model training, and evaluation.
- **Support for 22 Classes**: Classified characters include Abimanyu, Anoman, Arjuna, Bagong, Baladewa, Bima, Buta, Cakil, Durna, Dursasana, Duryudana, Gareng, Gatotkaca, Karna, Kresna, Nakula Sadewa, Patih Sabrang, Petruk, Puntadewa, Semar, Sengkuni, and Togog.

## Project Structure
```text
.
├── data/                                      # Dataset (Train, Test, Validation)
├── models/                                    # Saved model files (.h5)
│   └── non-augmented/                         # Models trained without augmentation
├── wayang_classifier_web/                     # Flask Web Application
│   ├── models/                                # Models used by the web app (model_1, 2, 3)
│   ├── static/                                # Static assets (uploads, css)
│   ├── templates/                             # HTML templates (index.html)
│   ├── app.py                                 # Flask entry point
│   └── requirements.txt                       # Web app dependencies
├── requirements.txt                           # Core project dependencies
├── Wayang_Classification_using_Ensemble_Learning_...ipynb # Training Notebook
└── README.md
```

## Requirements
- Python 3.10+
- TensorFlow 2.x
- Flask
- NumPy, Pandas, Scikit-learn, Matplotlib
- PIL (Pillow)

## Installation & Setup

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd JupyterProject
   ```

2. **Set up a virtual environment (recommended)**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   For core training and development:
   ```bash
   pip install -r requirements.txt
   ```
   For the web application specifically:
   ```bash
   pip install -r wayang_classifier_web/requirements.txt
   ```

## Usage

### Training and Evaluation
Open the Jupyter Notebook to explore the training process:
```bash
jupyter notebook "Wayang_Classification_using_Ensemble_Learning_With_InceptionV3,_EfficientNetB2,_and_ResNet101V2.ipynb"
```
The notebook includes:
- Data loading and augmentation using `ImageDataGenerator`.
- Transfer learning with InceptionV3, EfficientNetB2, and ResNet101V2.
- Ensemble logic implementation.
- Visualization of results (Accuracy/Loss curves, Confusion Matrix).

### Running the Web App
1. Navigate to the web app directory:
   ```bash
   cd wayang_classifier_web
   ```
2. Ensure the models (`model_1.h5`, `model_2.h5`, `model_3.h5`) are present in the `wayang_classifier_web/models/` directory.
3. Start the Flask server:
   ```bash
   python app.py
   ```
4. Open your browser and go to `http://127.0.0.1:5000`.

## Scripts
- `app.py`: Launches the Flask development server.
- Training Notebook: Contains all scripts for model training and performance metrics.

## Tests
- Manual verification can be done via the web app or the evaluation section of the notebook.

---
*Note: The dataset used in this project is stored in the `data/` directory, which is excluded from version control to keep the repository size manageable.*
