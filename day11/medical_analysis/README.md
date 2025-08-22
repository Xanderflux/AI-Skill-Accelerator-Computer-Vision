# Chest X-ray Pneumonia Classification: MLOps Pipeline

This project demonstrates a simple MLOps pipeline for binary classification of chest X-ray images into NORMAL and PNEUMONIA categories using a Convolutional Neural Network (CNN) based on VGG16. The pipeline incorporates:

- **MLFlow** for experiment tracking (parameters, metrics, models, and artifacts).
- **Prefect** for workflow orchestration.
- **FastAPI** for model serving via a REST API.
- **Docker** for containerizing the API.

The goal is to provide a reproducible, tracked, and deployable machine learning workflow.

## Prerequisites

Before you begin, ensure you have the following installed:

- **Python 3.9+**
- **Docker** (for containerizing the FastAPI service)
- **Git** (optional, for cloning the repository)
- **Kaggle API** (for downloading the dataset)
- A compatible environment (e.g., local machine, cloud VM, or Kaggle notebook)

## Project Structure

```
.
├── app.py                 # FastAPI application for model serving
├── Dockerfile             # Dockerfile for FastAPI service
├── requirements.txt       # Python dependencies
├── script.py             # Main script with MLFlow and Prefect pipeline
├── cnn_model.h5          # Trained model (generated after training)
├── sample_images.png     # Visualization of sample data (generated)
├── prediction.png        # Prediction visualization (generated)
├── probabilities.png     # Prediction probabilities chart (generated)
└── README.md             # This file
```

## Setup Instructions

### 1. Install Dependencies
Create a virtual environment and install the required Python packages:

```bash
pip install -r requirements.txt
```

The `requirements.txt` includes:
```
fastapi==0.115.0
uvicorn==0.30.6
tensorflow==2.17.0
numpy==1.26.4
pillow==10.4.0
mlflow==2.17.0
prefect==2.20.2
kaggle==1.6.17
matplotlib==3.9.2
```

### 2. Use Kaggle to download the chest X-ray dataset:

1. Download the dataset:
   ```bash
   kaggle datasets download -d paultimothymooney/chest-xray-pneumonia -p ./data
   unzip ./data/chest-xray-pneumonia.zip -d ./data
   ```
   This creates a `data/chest_xray` directory with `train`, `test`, and `val` subdirectories.

### 3. Start MLFlow Server
Run an MLFlow tracking server to log experiments:

```bash
mlflow server --host 0.0.0.0 --port 5000
```

- Access the MLFlow UI at `http://localhost:5000`.
- Ensure the tracking URI in `script.py` (`mlflow.set_tracking_uri("http://localhost:5000")`) matches your setup.

### 4. Configure Prefect
Initialize Prefect to manage workflows:

```bash
prefect init
```

Start the Prefect server (optional, for UI):

```bash
prefect server start
```

Access the Prefect UI at `http://localhost:4200`.

## Running the Training Pipeline

The `script.py` file orchestrates the training pipeline using Prefect tasks and logs results to MLFlow. To run it:

1. Ensure the dataset is in `./data/chest_xray` or update `train_dir`, `test_dir`, and `val_dir` in `script.py` to point to the correct paths.
2. Run the script:
   ```bash
   python script.py
   ```

### What Happens
- **Data Loading**: Loads and visualizes the dataset, logging sample images to MLFlow.
- **Data Preprocessing**: Applies data augmentation and preprocessing using `ImageDataGenerator`.
- **Model Training**: Trains a VGG16-based CNN, logging parameters, metrics, and the model to MLFlow.
- **Evaluation**: Evaluates the model on the test set, logging test metrics.
- **Prediction**: Makes a sample prediction on a validation image, logging visualizations.
- **Artifacts**: Saves the model (`cnn_model.h5`) and images (`sample_images.png`, `prediction.png`, `probabilities.png`) to MLFlow.

Check the MLFlow UI (`http://localhost:5000`) to view logged experiments, metrics, and artifacts.

## Building and Running the FastAPI Service
### Test the FastAPI app with uvicorn

```bash
uvicorn app:app --port 8000 --reload
```

### 1. Build the Docker Image
Ensure the `cnn_model.h5` file (generated from training) is in the project directory. Build the Docker image:

```bash
docker build -t pneumonia-api .
```

### 2. Run the FastAPI Container
Start the container, exposing port 8000:

```bash
docker run -p 8000:8000 pneumonia-api
```

The API will be available at `http://localhost:8000`.

## Testing the API

The FastAPI service exposes a `/predict` endpoint that accepts an image file and returns a JSON response with the predicted class and probabilities.

### Test with `curl`
1. Prepare a test image (e.g., `test_image.jpeg` from the dataset's `val` directory).
2. Send a POST request:

```bash
curl -X POST -F "file=@data/chest_xray/val/PNEUMONIA/person1950_bacteria_4881.jpeg" http://localhost:8000/predict
```

Example response:
```json
{
  "prediction": "PNEUMONIA",
  "probabilities": {
    "NORMAL": 0.123456,
    "PNEUMONIA": 0.876544
  }
}
```

### Test with Postman
1. Open Postman and create a new POST request to `http://localhost:8000/predict`.
2. In the "Body" tab, select `form-data`.
3. Add a key named `file`, set its type to `File`, and upload a test image.
4. Send the request and verify the response.

### Test with Python
Use the `requests` library to test the API:

```python
import requests

url = "http://localhost:8000/predict"
image_path = "/path/to/test_image.jpeg"  # Replace with actual path

with open(image_path, "rb") as f:
    response = requests.post(url, files={"file": f})

print(response.json())
```

## Expected Output
- **Training**: The pipeline trains the model for 10 epochs, logging metrics and artifacts to MLFlow. Test accuracy is printed (e.g., `Test accuracy: 85.50%`).
- **MLFlow**: View runs, metrics (train/val/test accuracy), and artifacts (model, images) in the MLFlow UI.
- **API**: The `/predict` endpoint returns a JSON object with the predicted class and probabilities for NORMAL and PNEUMONIA.

## Troubleshooting
- **Dataset Not Found**: Ensure the dataset is downloaded and paths in `script.py` are correct.
- **MLFlow Issues**: Verify the MLFlow server is running and the tracking URI is correct.
- **Docker Issues**: Check that `cnn_model.h5` is in the project directory and Docker is running.
- **API Errors**: Ensure the image is in RGB format and matches the expected size (224x224).

## Notes
- The validation set in the dataset is small. For better results, consider splitting the training set to create a larger validation set.
- Adjust `epochs`, `batch_size`, or model architecture in `script.py` for experimentation, and track changes in MLFlow.
- For production, secure the FastAPI service (e.g., add authentication) and use a managed MLFlow server.
