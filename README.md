# Heart Disease Prediction

A machine learning web application that predicts the likelihood of heart disease based on medical parameters. The project uses a K-Nearest Neighbors (KNN) classifier trained on the heart disease dataset to provide predictions through a user-friendly web interface.

## What It Does

This application takes 13 medical features as input and predicts whether a person is likely to have heart disease. The prediction is based on a trained KNN machine learning model that analyzes patterns in the data to classify patients into two categories:
- **Class 0**: No heart disease
- **Class 1**: Has heart disease

### Input Features

The model uses the following medical parameters:
- **age**: Age of the patient
- **sex**: Gender (1 = male, 0 = female)
- **chestpain**: Type of chest pain (0-3)
- **trestbps**: Resting blood pressure (in mm Hg)
- **chol**: Serum cholesterol (in mg/dl)
- **fbs**: Fasting blood sugar > 120 mg/dl (1 = true, 0 = false)
- **restecg**: Resting electrocardiographic results (0-2)
- **thalach**: Maximum heart rate achieved
- **exang**: Exercise induced angina (1 = yes, 0 = no)
- **oldpeak**: ST depression induced by exercise
- **slope**: Slope of peak exercise ST segment (0-2)
- **ca**: Number of major vessels colored by fluoroscopy (0-3)
- **thal**: Thalassemia (1-3)

## How It Works

### 1. Data Processing (`knn_model.py`)
- Loads the heart disease dataset from `heart.csv`
- Splits data into training (80%) and testing (20%) sets
- Applies StandardScaler to normalize features for better model performance
- Trains a K-Nearest Neighbors classifier with k=5 neighbors
- Evaluates the model using accuracy score and classification report
- Saves the trained model and scaler as `model.pkl` and `scaler.pkl`

### 2. Web Application (`app.py`)
- Built with Flask framework to provide a web interface
- Loads the pre-trained model and scaler
- Accepts user input through an HTML form
- Preprocesses input data using the same scaling transformation
- Makes predictions using the trained KNN model
- Displays results indicating heart disease likelihood

### 3. User Interface (`templates/index.html`)
- Provides an interactive form for entering medical parameters
- Displays prediction results in a user-friendly format
- Shows whether the user is likely or unlikely to have heart disease

## Setup Guide

### Prerequisites
- Python 3.7 or higher
- pip (Python package installer)

### Installation Steps

1. **Clone or download the repository**
   ```bash
   cd \Heart-Disease-Prediction
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv "venv"
   ```

3. **Activate the virtual environment**
   - On Windows (PowerShell):
     ```powershell
     .\venv\Scripts\Activate.ps1
     ```
   - On Windows (Command Prompt):
     ```cmd
     venv\Scripts\activate.bat
     ```
   - On macOS/Linux:
     ```bash
     source venv/bin/activate
     ```

4. **Install required dependencies**
   ```bash
   pip install -r requirement.txt
   ```

5. **Train the model** (First time setup)
   ```bash
   cd heartdisease
   python knn_model.py
   ```
   This will:
   - Train the KNN model on the heart disease dataset
   - Display accuracy metrics and classification report
   - Generate `model.pkl` and `scaler.pkl` files
   - Show a visualization of the model's performance

6. **Run the web application**
   ```bash
   python app.py
   ```

7. **Access the application**
   - Open your web browser
   - Navigate to: `http://127.0.0.1:5000/`
   - Fill in the medical parameters
   - Click submit to get the prediction

## Usage

1. Start the Flask web application:
   ```bash
   python app.py
   ```

2. Open your browser and go to `http://127.0.0.1:5000/`

3. Enter the required medical parameters in the form

4. Click the submit button to get the prediction

5. The application will display whether you are likely or unlikely to have heart disease

## Features

- **Machine Learning Model**: Uses K-Nearest Neighbors (KNN) algorithm for classification
- **Feature Scaling**: Implements StandardScaler for normalized predictions
- **Web Interface**: User-friendly Flask-based web application
- **Real-time Predictions**: Instant results based on input parameters
- **Model Persistence**: Saves trained model for quick predictions without retraining
- **Performance Visualization**: Generates classification report with accuracy metrics

## Project Structure

```
Heart-Disease-Prediction/
│
├── heartdisease/
│   ├── app.py              # Flask web application
│   ├── knn_model.py         # Model training script
│   ├── heart.csv            # Heart disease dataset
│   ├── templates/
│   │   └── index.html       # Web interface template
│   ├── model.pkl            # Trained KNN model (generated)
│   └── scaler.pkl           # Fitted scaler (generated)
│
├── requirement.txt          # Python dependencies
└── README.md               # Project documentation
```

## Technologies Used

- **Python 3.x**: Core programming language
- **Flask**: Web framework for the application
- **scikit-learn**: Machine learning library (KNN, StandardScaler)
- **pandas**: Data manipulation and analysis
- **numpy**: Numerical computing
- **joblib**: Model serialization
- **matplotlib**: Data visualization (for training metrics)

## Model Performance

The KNN model is evaluated using:
- **Accuracy Score**: Overall prediction accuracy
- **Classification Report**: Precision, recall, and F1-score for both classes
- **Train-Test Split**: 80% training, 20% testing with random_state=42

## Troubleshooting

- **ModuleNotFoundError**: Ensure all dependencies are installed using `pip install -r requirement.txt`
- **FileNotFoundError for .pkl files**: Run `knn_model.py` first to generate the model files
- **Port already in use**: Change the port in `app.py` or stop the process using port 5000
- **Virtual environment issues**: Make sure the virtual environment is activated before running commands

## Future Enhancements

- Add more machine learning models (Random Forest, SVM, Neural Networks)
- Implement model comparison and ensemble methods
- Add data visualization for input parameters
- Include confidence scores with predictions
- Deploy to cloud platforms (Heroku, AWS, Azure)
- Add user authentication and prediction history
