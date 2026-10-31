
import pandas as pd
import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report


# 1. Create dataset
data = {
    "CGPA": [
        8.5, 7.8, 6.2, 9.1, 5.8,
        7.2, 6.5, 8.9, 5.5, 7.5,
        6.8, 8.2, 5.9, 9.3, 7.0,
        6.0, 8.7, 6.4, 7.9, 5.7,
        8.0, 6.1, 8.8, 7.3, 5.6
    ],
    "Aptitude_Score": [
        85, 78, 55, 92, 48,
        70, 60, 88, 45, 75,
        65, 80, 52, 95, 68,
        50, 90, 58, 82, 47,
        77, 54, 91, 72, 42
    ],
    "Technical_Skills_Score": [
        90, 82, 60, 95, 50,
        75, 55, 91, 48, 80,
        62, 85, 55, 98, 72,
        52, 88, 63, 79, 45,
        83, 58, 93, 70, 49
    ],
    "Placement_Status": [
        "Placed", "Placed", "Not Placed", "Placed", "Not Placed",
        "Placed", "Not Placed", "Placed", "Not Placed", "Placed",
        "Not Placed", "Placed", "Not Placed", "Placed", "Placed",
        "Not Placed", "Placed", "Not Placed", "Placed", "Not Placed",
        "Placed", "Not Placed", "Placed", "Placed", "Not Placed"
    ]
}

df = pd.DataFrame(data)

# Save dataset to CSV
df.to_csv("student_placement.csv", index=False)

print("Dataset created successfully!")
print(df)

# 2. Separate features and target
X = df[
    ["CGPA", "Aptitude_Score", "Technical_Skills_Score"]
]
y = df["Placement_Status"]

# 3. Split the dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))

# 4. Configure MLflow
mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("Student Placement Prediction")

results = []

# 5. Train two experiments
for depth in [2, 4]:

    with mlflow.start_run(
        run_name=f"DecisionTree_depth_{depth}"
    ):

        model = DecisionTreeClassifier(
            max_depth=depth,
            random_state=42
        )

        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)

        # Log parameters
        mlflow.log_param(
            "model_type", "DecisionTreeClassifier"
        )
        mlflow.log_param("max_depth", depth)
        mlflow.log_param("test_size", 0.2)
        mlflow.log_param("random_state", 42)

        # Log metric
        mlflow.log_metric("accuracy", accuracy)

        # Log trained model
        mlflow.sklearn.log_model(
            sk_model=model,
            name="student_placement_model",
            skops_trusted_types=[
                "sklearn.tree._tree.Tree"
            ]
        )

        run_id = mlflow.active_run().info.run_id

        results.append({
            "Max_Depth": depth,
            "Accuracy": accuracy,
            "Run_ID": run_id
        })

        print(f"\nDecision Tree max_depth={depth}")
        print(f"Accuracy: {accuracy:.4f}")

        print(classification_report(
            y_test,
            y_pred,
            zero_division=0
        ))

# 6. Compare experiments
results_df = pd.DataFrame(results)

print("\nEXPERIMENT COMPARISON")
print(results_df.to_string(index=False))

results_df.to_csv(
    "experiment_comparison.csv",
    index=False
)

best_result = results_df.loc[
    results_df["Accuracy"].idxmax()
]

print("\nBEST MODEL")
print("Max depth:", int(best_result["Max_Depth"]))
print("Accuracy:", round(best_result["Accuracy"], 4))

# 7. Predict for a new student using the best depth
new_student = pd.DataFrame({
    "CGPA": [8.2],
    "Aptitude_Score": [85],
    "Technical_Skills_Score": [80]
})

final_model = DecisionTreeClassifier(
    max_depth=int(best_result["Max_Depth"]),
    random_state=42
)

final_model.fit(X, y)

prediction = final_model.predict(new_student)

print("\nNEW STUDENT PREDICTION")
print(new_student)
print("Placement Status:", prediction[0])

print("\nProject completed successfully!")
