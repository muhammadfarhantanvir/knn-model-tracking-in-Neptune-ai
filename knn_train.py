# Import Necessary Libraries
import neptune
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from sklearn import datasets
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Initialize Neptune Run
run = neptune.init_run(
    project="tanvir/knn",  # Format: "workspace_name/project_name"
    api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiI4NGEwNDdmZS03ZDM0LTQ0NjYtYmQ0OS1jYWFiODM2OTA3MDcifQ==",  # Your Neptune API token
    name="knn",  # Experiment name
    tags=["KNN", "Iris"]  # Experiment tags
)

# Load Sample Dataset (Iris Dataset)
iris = datasets.load_iris()
X = iris.data  # Features
y = iris.target  # Labels

# Prepare Data: Split into Training and Testing Sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize Features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Log Parameters
run["test_size"] = 0.2
run["random_state"] = 42
run["k_value"] = 5  # Default k value, can be changed

# Build KNN Model
k = 5
knn = KNeighborsClassifier(n_neighbors=k)

# Train the Model
knn.fit(X_train, y_train)

# Make Predictions
y_pred = knn.predict(X_test)
y_pred_proba = knn.predict_proba(X_test)  # Predict probabilities for AUC

# Evaluate the Model
accuracy = metrics.accuracy_score(y_test, y_pred)
run["accuracy"] = accuracy  # Log Accuracy

# --- AUC Plot ---
auc = metrics.roc_auc_score(y_test, y_pred_proba, multi_class='ovr')
run["auc"] = auc  # Log AUC

# Since Neptune doesn't directly support plotting AUC, we'll save the plot locally and upload it
plt.figure(figsize=(8, 6))
plt.plot(y_pred_proba[:, 1], label='Class 1 vs Rest')
plt.xlabel('Sample Index')
plt.ylabel('Predicted Probability')
plt.title('AUC Plot (Class 1 vs Rest)')
plt.legend()
plt.savefig('auc_plot.png')
run["auc_plot"].upload("auc_plot.png")  # Upload AUC Plot with updated syntax

# --- Confusion Matrix ---
cm = metrics.confusion_matrix(y_test, y_pred)
run["confusion_matrix"] = cm.tolist()  # Log Confusion Matrix as a list

# Plot Confusion Matrix using Seaborn
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, cmap='Blues')
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix')
plt.savefig('confusion_matrix.png')
run["confusion_matrix_plot"].upload("confusion_matrix.png")  # Upload Confusion Matrix Plot with updated syntax

# Log Model (Optional but recommended for reproducibility)
import pickle
with open('model.pkl', 'wb') as f:
    pickle.dump(knn, f)
run["model"].upload("model.pkl")  # Upload the pickled model with updated syntax