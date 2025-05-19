# Importing necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# Load the dataset from local CSV file
data = pd.read_csv('heartdisease.csv')  # Use the path to your local CSV file

# Preprocessing - Handling missing values (if any)
data = data.replace('?', pd.NA)  # Replace any missing values marked as '?'
data = data.dropna()  # Drop rows with missing values

# Features and Target
X = data.drop('target', axis=1)  # Features
y = data['target']  # Target variable

# Splitting the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature Scaling (important for KNN)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Training KNN Classifier
knn = KNeighborsClassifier(n_neighbors=5)  # You can experiment with n_neighbors
knn.fit(X_train, y_train)

# Evaluating the model on test set (optional, you can skip this when just predicting)
y_pred = knn.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy * 100:.2f}%\n")

# Function to predict heart disease for user input and store it in the same CSV file
def predict_heart_disease():
    print("\nPlease enter the following information:")

    # Asking user for input
    age = float(input("Age: "))
    sex = float(input("Sex (1 = male, 0 = female): "))
    cp = float(input("Chest Pain Type (1-4): "))
    trestbps = float(input("Resting Blood Pressure: "))
    chol = float(input("Cholesterol Level: "))
    fbs = float(input("Fasting Blood Sugar (1 = true, 0 = false): "))
    restecg = float(input("Resting Electrocardiographic Results (0-2): "))
    thalach = float(input("Maximum Heart Rate Achieved: "))
    exang = float(input("Exercise Induced Angina (1 = yes, 0 = no): "))
    oldpeak = float(input("Oldpeak (depression induced by exercise relative to rest): "))
    slope = float(input("Slope of the Peak Exercise ST Segment (1-3): "))
    ca = float(input("Number of Major Vessels Colored by Fluoroscopy (0-4): "))
    thal = float(input("Thalassemia (3 = normal, 6 = fixed defect, 7 = reversible defect): "))

    # Create a DataFrame for the input
    user_input = pd.DataFrame([[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]],
                              columns=['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal'])

    # Scaling the user input (important for KNN)
    user_input_scaled = scaler.transform(user_input)

    # Predict whether the user has heart disease or not
    prediction = knn.predict(user_input_scaled)

    # Output the prediction result
    if prediction == 1:
        print("\nThe prediction is: You have heart disease.")
    else:
        print("\nThe prediction is: You do not have heart disease.")

    # Save the prediction to the same CSV file
    save_prediction_to_same_csv(age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal, prediction[0])

# Function to save prediction data to the same CSV file
def save_prediction_to_same_csv(age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal, prediction):
    # Creating a DataFrame for the new prediction
    new_data = pd.DataFrame([[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal, prediction]],
                            columns=['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'target'])

    # Append new data to the same CSV file (heart_disease.csv)
    new_data.to_csv('heartdisease.csv', mode='a', header=False, index=False)

    print("\nPrediction data has been saved to 'heartdisease.csv'.")

# Function to retrain the model with the new data from 'heart_disease.csv'
def retrain_model():
    # Load the new data with predictions
    new_data = pd.read_csv('heartdisease.csv')

    # Separate features and target
    X_new = new_data.drop('target', axis=1)
    y_new = new_data['target']

    # Split the new data into training and test sets
    X_train_new, X_test_new, y_train_new, y_test_new = train_test_split(X_new, y_new, test_size=0.2, random_state=42)

    # Scale the new data
    X_train_new = scaler.fit_transform(X_train_new)
    X_test_new = scaler.transform(X_test_new)

    # Retrain the KNN classifier with the new data
    knn.fit(X_train_new, y_train_new)

    # Evaluate the new model
    y_pred_new = knn.predict(X_test_new)
    accuracy_new = accuracy_score(y_test_new, y_pred_new)
    print(f"New Model Accuracy with Additional Data: {accuracy_new * 100:.2f}%")

# Call the function to get prediction and store it in the same file
predict_heart_disease()

# Optionally retrain the model with new data from 'heart_disease.csv'
# retrain_model()  # Uncomment to retrain the model after predictions are saved
