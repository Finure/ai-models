# Finure AI Models

## 1. Overview

Finure ai-models provides the custom model training and test code for the Finure platform. It includes Python code (`model.py`) based on scikit-learn framework for training and evaluating a logistic regression model to predict credit card approval, using datasets from Google Cloud Storage and then saves and uploads the model to Google Cloud Storage to be used in production for inference using KServe and Knative. The Dockerfile, requirements.txt, along with the model code are used by Argo Workflows to build and run the model training job in a Kubernetes cluster as part of the Finure ML pipeline.

## 2. Features
- **Model Training:** Trains a logistic regression model based on scikit-learn framework for credit card approval prediction
- **Data Validation & Preprocessing:** Handles missing values, scaling, encoding, and normalization of input data
- **Stratified Split & Evaluation:** Performs stratified train/test split and evaluates accuracy, precision, recall, F1, ROC-AUC, and confusion matrix
- **Model Export:** Saves trained model as a joblib file and uploads to GCS
- **Sample Prediction:** Runs a hardcoded sample through the trained model for quick testing
- **Configurable via Environment:** Reads dataset and output paths from environment variables for flexible deployment
- **Kubernetes Ready:** Built part of the Finure ML pipeline, so the training job can be containerized and run in a k8s cluster using Argo Events and Argo Workflows 

## 3. Model Performance:
Evaluation Metrics on Test Set:
| Metric     | Value  |
|------------|--------|
| Accuracy   | 0.9302 |
| Precision  | 0.7971 |
| Recall     | 0.7971 |
| F1-Score   | 0.7971 |
| ROC-AUC    | 0.9831 |


Confusion Matrix:
|            | Predicted 0 | Predicted 1 |
|------------|-------------|-------------|
| Actual 0   | 318         | 14          |
| Actual 1   | 14          | 55          |

Classification Report:
| Class            | Precision | Recall | F1-Score | Support |
|------------------|-----------|--------|----------|---------|
| 0                | 0.96      | 0.96   | 0.96     | 332     |
| 1                | 0.80      | 0.80   | 0.80     | 69      |
| **Accuracy**     |           |        | 0.93     | 401     |
| **Macro Avg**    | 0.88      | 0.88   | 0.88     | 401     |
| **Weighted Avg** | 0.93      | 0.93   | 0.93     | 401     |

## 4. Prerequisites
- Kubernetes cluster bootstrapped ([Finure Terraform](https://github.com/finure/terraform))
- Infrastructure setup via Flux ([Finure Kubernetes](https://github.com/finure/kubernetes))

If running locally for development/testing:
- Google Cloud Storage bucket and credentials (optional)
- Python 3.12+
- Docker
- Required Python packages (see `requirements.txt`)

## 5. File Structure
```
ai-models/
├── Dockerfile           # Container build file for model training
├── requirements.txt     # Python dependencies
├── model.py             # Model training and inference script
├── .dockerignore        # Docker ignore rules
├── .gitignore           # Git ignore rules
├── README.md            # Project documentation
```

## 6. How to Run Manually

> **Note:** Manual execution is for development/testing only. Production use is via containerized jobs in Kubernetes.

1. Install Python dependencies:
	```bash
	pip install -r requirements.txt
	```
2. Set required environment variables:
	- `BUCKET`: GCS bucket containing the dataset
	- `OUTFILE`: Path to the dataset in the bucket
	- (Optional) `DATA_PATH`: Local path to save the dataset
3. Run the model training script:
	```bash
	python model.py
	```
	The trained model will be saved locally and uploaded to GCS.

## 7. Deployment

This model code is designed to be run as a containerized job in a Kubernetes cluster as part of the Finure ML pipeline. Use the provided Dockerfile to build the image and deploy using your cluster's orchestration tools.

## Additional Information

This repo is primarily designed to be used in the Finure project. While the model can be adapted for other use cases, it is recommended to use it as part of the Finure platform for full functionality and support.