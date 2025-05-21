import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib

data_frame = pd.read_csv('heart.csv')

# Displaying dataframe info
# print(data_frame.head())
# print(data_frame.info())
# print(data_frame.describe())

X_input = data_frame.drop("target",axis = 1) #input without target
Y_output = data_frame["target"] #only target value

# Train set 80% and test set 20%
X_train, X_test, y_train, y_test  = train_test_split(X_input,Y_output, test_size = 0.2, random_state = 42)

# Scaling mean in the form of 0 and standard deviation in the form of 1
# fit_transform(): Learns the scaling parameters (mean & std) from the data and applies the scaling.
# transform(): Uses previously learned scaling parameters to scale new data without learning again.
scaler = StandardScaler()
X_train_scale = scaler.fit_transform(X_train)
X_test_scale = scaler.transform(X_test)

# KNN model
# n_neighbour = 31: To classify new point , the neighbour will  look at the 31 nearest data
knn = KNeighborsClassifier(n_neighbors=31)
knn.fit(X_train_scale, y_train)

y_pred = knn.predict(X_test_scale)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# To check
# def predict_heart_disease(model, scaler, input_data):
#     """
#     model: trained KNN model
#     scaler: the StandardScaler used to scale training data
#     input_data: list or array of input feature values in correct order
    
#     Returns prediction 0 or 1
#     """
#     # Convert input to 2D array for scaler/model (1 sample, n features)
#     input_array = [input_data]
    
#     # Scale the input the same way training data was scaled
#     input_scaled = scaler.transform(input_array)
#     # Predict
#     prediction = model.predict(input_scaled)
#     print("Prediction:",prediction)
#     return prediction[0]

# # ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal'](IN This order)

# new_patient = [55, 1, 0, 130, 250, 0, 1, 150, 0, 1.5, 2, 0, 2]

# result = predict_heart_disease(knn, scaler, new_patient)

# if result == 1:
#     print("Prediction: You may have heart disease.")
# else:
#     print("Prediction: You are unlikely to have heart disease.")

joblib.dump(knn, 'model.pkl')
joblib.dump(scaler, 'scaler.pkl')