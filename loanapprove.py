
import pandas as pd
import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

data = {
    "Applicant Income": [
        25000, 30000, 45000, 50000, 60000,
        22000, 35000, 70000, 80000, 40000,
        28000, 90000, 55000, 32000, 75000,
        48000, 20000, 85000, 65000, 38000
    ],
    "Credit Score": [
        550, 600, 700, 720, 750,
        520, 650, 780, 800, 680,
        580, 820, 710, 610, 790,
        690, 500, 810, 760, 640
    ],
    "Loan Amount": [
        200000, 180000, 150000, 200000, 250000,
        220000, 170000, 180000, 200000, 160000,
        210000, 150000, 190000, 230000, 170000,
        180000, 240000, 160000, 200000, 190000
    ],
    "Loan Status": [
        "Rejected", "Rejected", "Approved", "Approved",
        "Approved", "Rejected", "Approved", "Approved",
        "Approved", "Approved", "Rejected", "Approved",
        "Approved", "Rejected", "Approved", "Approved",
        "Rejected", "Approved", "Approved", "Rejected"
    ]
}

df = pd.DataFrame(data)
df.to_csv("loan_approval.csv", index=False)

print("\nLoan Approval Dataset:")
print(df.head())

X = df[["Applicant Income", "Credit Score", "Loan Amount"]]
y = df["Loan Status"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("Loan Approval Prediction")

results = []
best_accuracy = -1
best_model = None
best_depth = None

for depth in [2, 4]:
    with mlflow.start_run(run_name=f"Decision_Tree_depth_{depth}"):

        model = DecisionTreeClassifier(
            max_depth=depth,
            random_state=42
        )

        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)

        mlflow.log_param("model_type", "DecisionTreeClassifier")
        mlflow.log_param("max_depth", depth)
        mlflow.log_param("test_size", 0.2)
        mlflow.log_param("random_state", 42)
        mlflow.log_metric("accuracy", float(accuracy))

        mlflow.sklearn.log_model(
            sk_model=model,
            name="loan_approval_model",
            skops_trusted_types=["sklearn.tree._tree.Tree"]
        )

        results.append({
            "max_depth": depth,
            "accuracy": accuracy
        })

        print(f"\nDecision Tree (max_depth={depth})")
        print(f"Accuracy: {accuracy:.2%}")

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_model = model
            best_depth = depth

results_df = pd.DataFrame(results)
results_df.to_csv("experiment_comparison.csv", index=False)

print("\nModel Comparison:")
print(results_df.to_string(index=False))

print(f"\nBest Model: Decision Tree (max_depth={best_depth})")
print(f"Best Accuracy: {best_accuracy:.2%}")

new_applicant = pd.DataFrame({
    "Applicant Income": [50000],
    "Credit Score": [720],
    "Loan Amount": [180000]
})

prediction = best_model.predict(new_applicant)[0]

print("\nNew Applicant Details:")
print(new_applicant.to_string(index=False))
print("Predicted Loan Status:", prediction)

print("\nLoan Approval Prediction Completed Successfully!")
