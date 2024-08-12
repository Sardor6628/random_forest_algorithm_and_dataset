import os
import glob
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import joblib

# Define the path to the dataset
data_path = "dataset/normalized_data/"


# Load the datasets from both SIH and SIM folders
def load_data():
    all_files = glob.glob(os.path.join(data_path, "**/*.csv"), recursive=True)
    data_list = []

    for file in all_files:
        # Determine the label from the file name
        if "dynamic_0" in file:
            label = 0
        elif "dynamic_1" in file:
            label = 1
        elif "dynamic_2" in file:
            label = 2
        elif "dynamic_3" in file:
            label = 3
        elif "dynamic_4" in file:
            label = 4
        elif "dynamic_5" in file:
            label = 5
        else:
            continue

        # Load the data and append the label
        df = pd.read_csv(file)
        df['label'] = label
        data_list.append(df)

    # Concatenate all the data into a single DataFrame
    return pd.concat(data_list, ignore_index=True)


# Load the data
data = load_data()

# Specify the features and labels
feature_columns = [
    "lt_hip_sagittal", "lt_hip_frontal", "lt_hip_transe",
    "lt_knee_sagittal", "lt_knee_frontal", "lt_knee_transe",
    "lt_ank_sagittal", "lt_ank_frontal", "lt_ank_transe",
    "rt_hip_sagittal", "rt_hip_frontal", "rt_hip_transe",
    "rt_knee_sagittal", "rt_knee_frontal", "rt_knee_transe",
    "rt_ank_sagittal", "rt_ank_frontal", "rt_ank_transe",
    "lt_plv_sagittal", "lt_plv_frontal", "lt_plv_transe",
    "rt_plv_sagittal", "rt_plv_frontal", "rt_plv_transe"
]
X = data[feature_columns]
y = data['label']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Create and train the Random Forest classifier
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
rf_classifier.fit(X_train, y_train)

# Make predictions on the test set
y_pred = rf_classifier.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
# Explicitly define all possible labels
labels = [0, 1, 2, 3, 4, 5]
target_names = [
    "Acceptable squat posture",
    "Knees flaring out",
    "Knees collapsing inward",
    "Toe raising off the ground",
    "Excessive hip flexion",
    "Excessive posterior pelvic tilt"
]

report = classification_report(y_test, y_pred, labels=labels, target_names=target_names, zero_division=0)

# Print accuracy and classification report
print(f"Accuracy: {accuracy:.2f}")
print("Classification Report:")
print(report)

# Save the trained model to a file
joblib.dump(rf_classifier, 'random_forest_model.pkl')

# Save the classification report to a text file
with open('classification_report.txt', 'w') as f:
    f.write(f"Accuracy: {accuracy:.2f}\n")
    f.write("Classification Report:\n")
    f.write(report)
