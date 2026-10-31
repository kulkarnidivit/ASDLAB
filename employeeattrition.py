

#Develop a machine learning classification model to predict whether an employee will Stay or Leave the organization based on Years of Experience, Monthly Salary, and Job Satisfaction Score.
# Create a CSV dataset named employee_attrition.csv.
# Load the dataset and separate input features and target.
# Split the dataset into training and testing sets.
# Train a Decision Tree Classifier and calculate its accuracy.
# Use MLflow to log model parameters, accuracy metrics, and the trained model.
# Compare two experiments by changing the max_depth parameter











import pandas as pd
import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

data = {
    "Years of Experience": [
        1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
        2, 4, 6, 8, 10, 3, 5, 7, 9, 12,
        1, 3, 5, 7, 11, 2, 6, 8, 13, 15
    ],
    "Monthly Salary": [
        25000, 30000, 35000, 40000, 45000,
        50000, 55000, 60000, 65000, 70000,
        28000, 38000, 48000, 58000, 75000,
        32000, 42000, 52000, 62000, 80000,
        22000, 33000, 47000, 57000, 85000,
        27000, 49000, 63000, 90000, 100000
    ],
    "Job Satisfaction Score": [
        2, 3, 3, 4, 4, 5, 5, 4, 5, 5,
        2, 3, 4, 4, 5, 2, 3, 4, 5, 5,
        1, 2, 3, 4, 5, 2, 3, 4, 5, 5
    ],
    "Attrition": [
        "Leave", "Leave", "Stay", "Stay", "Stay",
        "Stay", "Stay", "Stay", "Stay", "Stay",
        "Leave", "Leave", "Stay", "Stay", "Stay",
        "Leave", "Stay", "Stay", "Stay", "Stay",
        "Leave", "Leave", "Leave", "Stay", "Stay",
        "Leave", "Stay", "Stay", "Stay", "Stay"
    ]
}

df = pd.DataFrame(data)
df.to_csv("employee_attrition.csv", index=False)

print("Dataset created successfully: employee_attrition.csv")
print("\nEmployee Attrition Dataset:")
print(df.head())
print("\nTotal records:", len(df))

df = pd.read_csv("employee_attrition.csv")

X = df[[
    "Years of Experience",
    "Monthly Salary",
    "Job Satisfaction Score"
]]

y = df["Attrition"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nTraining records:", len(X_train))
print("Testing records:", len(X_test))

mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("Employee Attrition Prediction")

results = []
trained_models = {}

for depth in [2, 4]:
    with mlflow.start_run(run_name=f"Decision_Tree_depth_{depth}"):
        model = DecisionTreeClassifier(
            max_depth=depth,
            random_state=42
        )

        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)

        mlflow.log_param("model_type", "Decision Tree")
        mlflow.log_param("max_depth", depth)
        mlflow.log_param("test_size", 0.2)
        mlflow.log_param("random_state", 42)
        mlflow.log_metric("accuracy", accuracy)

        mlflow.sklearn.log_model(
            sk_model=model,
            name="employee_attrition_model",
            skops_trusted_types=["sklearn.tree._tree.Tree"]
        )

        results.append({
            "Model": "Decision Tree",
            "Max Depth": depth,
            "Accuracy": accuracy
        })

        trained_models[depth] = model

        print(f"\nDecision Tree (max_depth={depth})")
        print(f"Accuracy: {accuracy * 100:.2f}%")
        print("\nConfusion Matrix:")
        print(confusion_matrix(y_test, y_pred, labels=["Stay", "Leave"]))
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, zero_division=0))

comparison = pd.DataFrame(results)
comparison.to_csv("experiment_comparison.csv", index=False)

print("\nExperiment Comparison:")
print(comparison.to_string(index=False))

best_depth = int(
    comparison.loc[comparison["Accuracy"].idxmax(), "Max Depth"]
)

best_model = trained_models[best_depth]

print(f"\nBest Model: Decision Tree (max_depth={best_depth})")
print(f"Best Accuracy: {comparison['Accuracy'].max() * 100:.2f}%")

new_employee = pd.DataFrame([{
    "Years of Experience": 2,
    "Monthly Salary": 30000,
    "Job Satisfaction Score": 2
}])

prediction = best_model.predict(new_employee)[0]

print("\nNew Employee Details:")
print(new_employee.to_string(index=False))
print("Predicted Attrition:", prediction)

print("\nEmployee Attrition Classification Completed Successfully!")
