from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import joblib
import os

# 1 - load
iris = load_iris()
X = iris.data
y = iris.target

print("Number of samples:", X.shape[0])
print("Number of features:", X.shape[1])
print("Feature names:", iris.feature_names)
print("Target/class names:", [str(name) for name in iris.target_names])

# 2 - create training and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)

print("\nTraining samples:", X_train.shape[0])
print("Test samples:", X_test.shape[0])

# 3 - choose and train a classifier
model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)

# 4 - make predictions
y_pred = model.predict(X_test)

print("\nActual:   ", [str(iris.target_names[i]) for i in y_test[:10]])
print("Predicted:", [str(iris.target_names[i]) for i in y_pred[:10]])

# 5 - evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print("\nTest Accuracy:", accuracy)

cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(cm)

disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=iris.target_names)
disp.plot(cmap="Blues")
plt.show()

# 6 - save the trained model
joblib.dump(model, "model.joblib")
print("\nmodel.joblib saved:", os.path.exists("model.joblib"))

# 7 - load and reuse the model
loaded_model = joblib.load("model.joblib")
y_pred_loaded = loaded_model.predict(X_test)
accuracy_loaded = accuracy_score(y_test, y_pred_loaded)

print("\nOriginal model accuracy:", accuracy)
print("Loaded model accuracy:  ", accuracy_loaded)
