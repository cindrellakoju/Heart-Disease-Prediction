import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib
import matplotlib.pyplot as plt

# 1. Load data
data_frame = pd.read_csv('heart.csv')

# 2. Prepare input and target
X_input = data_frame.drop("target", axis=1)
Y_output = data_frame["target"]

# 3. Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_input, Y_output, test_size=0.2, random_state=42)

# 4. Feature scaling
scaler = StandardScaler()
X_train_scale = scaler.fit_transform(X_train)
X_test_scale = scaler.transform(X_test)

# 5. Train KNN model
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train_scale, y_train)

# 6. Predict and evaluate
y_pred = knn.predict(X_test_scale)
acc = accuracy_score(y_test, y_pred)
report_dict = classification_report(y_test, y_pred, output_dict=True)
print("Accuracy:", acc)
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# 7. Plot classification report
classes = ['0', '1']
metrics = ['precision', 'recall', 'f1-score']

fig, ax = plt.subplots(figsize=(8, 4))
for i, metric in enumerate(metrics):
    values = [report_dict[c][metric] for c in classes]
    ax.bar([x + i*0.25 for x in range(len(classes))], values, width=0.25, label=metric)

ax.set_xticks([r + 0.25 for r in range(len(classes))])
ax.set_xticklabels(['Class 0 (No HD)', 'Class 1 (Has HD)'])
ax.set_ylim(0, 1.1)
ax.set_title(f"Classification Report (Accuracy = {acc:.2f})")
ax.set_ylabel("Score")
ax.legend()
plt.tight_layout()
plt.show()

# 8. Save model and scaler
joblib.dump(knn, 'model.pkl')
joblib.dump(scaler, 'scaler.pkl')
