# Maize Leaf Disease Detection

This repository contains a deep learning-based image classification project for detecting maize leaf disease from leaf images. The project includes a training script, a trained Keras model, and a simple application interface for making predictions from uploaded maize leaf images.

This project was developed as an applied computer vision and deep learning project and is maintained as part of my machine learning project portfolio.

## Project Overview

Maize is an important cereal crop cultivated widely across different regions of the world. However, maize production can be affected by several leaf diseases, which may reduce crop yield and affect food security if not detected early.

Computer vision and deep learning methods can support plant disease detection by learning visual patterns from leaf images. In this project, a convolutional neural network-based workflow is used to classify maize leaf images into disease categories.

The goal of this project is to demonstrate how image classification can be applied to agricultural disease detection using Python, TensorFlow/Keras, and a simple prediction interface.

## Objectives

The main objectives of this project are to:

- build a deep learning model for maize leaf disease classification
- train the model using maize leaf image data
- save the trained model for reuse
- create a simple application for image-based prediction
- demonstrate an end-to-end computer vision workflow for plant disease detection

## Repository Structure

```text
Maize-Leaf-Disease-Detection/
├── app.py
├── train_model.py
├── maize_leaf_disease_model.keras
├── maize_leaf_dataset/
├── README.md
├── requirements.txt
├── .gitignore
└── LICENSE
```

## Files Description

### `app.py`

This file contains the application code for loading the trained model and making predictions from maize leaf images. It allows users to upload an image and receive a predicted disease class.

### `train_model.py`

This file contains the model training workflow. It is used to load the image dataset, preprocess the images, train the deep learning model, and save the trained model.

### `maize_leaf_disease_model.keras`

This is the saved trained Keras model. It can be loaded directly for prediction without retraining the model.

### `maize_leaf_dataset/`

This folder contains the maize leaf image dataset used for training and testing the model.

## Methodology

The project follows a standard computer vision workflow:

### 1. Dataset Preparation

The maize leaf images are organized into class-specific folders. Each folder represents a disease category or leaf condition.

The dataset is loaded and prepared for model training using image preprocessing techniques such as resizing, normalization, and batching.

### 2. Image Preprocessing

Images are resized to a fixed input shape suitable for the deep learning model. Pixel values are normalized to improve model training stability.

Common preprocessing steps include:

- image resizing
- pixel normalization
- train-validation splitting
- batch loading

### 3. Model Training

A deep learning model is trained on the maize leaf image dataset. The model learns visual features that distinguish different maize leaf conditions.

The trained model is then saved in Keras format for later use.

### 4. Model Prediction

The saved model is loaded by the application. When a user uploads a maize leaf image, the image is preprocessed and passed into the model for classification.

The application returns the predicted class based on the model output.

## How to Run the Project

Clone the repository:

```bash
git clone https://github.com/CodeeSam/Maize-Leaf-Disease-Detection.git
cd Maize-Leaf-Disease-Detection
```

Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Running the Application

To launch the application, run:

```bash
streamlit run app.py
```

After running the command, Streamlit will open the application in your browser. You can then upload a maize leaf image and obtain a model prediction.

## Training the Model

To retrain the model, run:

```bash
python train_model.py
```

After training, the model will be saved as:

```text
maize_leaf_disease_model.keras
```

## Requirements

The main Python packages used in this project include:

```text
tensorflow
numpy
pillow
streamlit
matplotlib
scikit-learn
```

A typical `requirements.txt` file may include:

```text
tensorflow
numpy
pillow
streamlit
matplotlib
scikit-learn
```

## Example Workflow

The general workflow is:

```text
Image Dataset → Preprocessing → Model Training → Saved Keras Model → Streamlit App → Image Prediction
```

## Project Note

This repository represents an applied deep learning project in computer vision. It is intended to demonstrate the use of image classification for maize leaf disease detection.

The project should be interpreted as a demonstration and learning-based application, not as a clinically or agriculturally validated diagnostic tool.

## Limitations

Some limitations of this project include:

- The model performance depends on the size and quality of the training dataset.
- The model may not generalize well to images captured under very different lighting, background, or field conditions.
- The project does not include large-scale external validation.
- The app is designed for demonstration and educational purposes.
- Field-level deployment would require further validation using diverse real-world maize leaf images.

## Future Improvements

Possible future improvements include:

- adding more maize leaf images from diverse field conditions
- performing stronger train-validation-test splitting
- evaluating the model using precision, recall, F1-score, and confusion matrix
- adding data augmentation to improve generalization
- testing transfer learning models such as MobileNetV2, EfficientNet, or ResNet
- deploying the application online
- adding model confidence scores
- improving the user interface
- adding support for multiple crop disease detection

## Applications

This type of project can be useful as a starting point for:

- agricultural AI applications
- plant disease image classification
- computer vision learning projects
- crop disease monitoring prototypes
- deep learning deployment practice

## Author

**Samson Ayorinde Oni**  
Machine Learning | Deep Learning | Biotechnology  

## License

This repository is released under the MIT License.
