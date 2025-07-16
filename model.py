import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import classification_report, roc_auc_score, accuracy_score
from imblearn.over_sampling import SMOTE
import pickle

# Load dataset
df = pd.read_csv("diabetes_data.csv")

# Features and target
X = df.drop(columns=["Diabetes_binary"])
y = df["Diabetes_binary"]

# Handle class imbalance
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_resampled)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_resampled, test_size=0.2, random_state=42)

# Models
lr = LogisticRegression(max_iter=1000)
xgb = XGBClassifier(use_label_encoder=False, eval_metric='logloss')

# Ensemble Voting Classifier
voting_model = VotingClassifier(estimators=[("lr", lr), ("xgb", xgb)], voting="soft")
voting_model.fit(X_train, y_train)

# Make predictions
y_pred = voting_model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy * 100:.2f}%")  # Print the accuracy in percentage

# Print classification report and ROC-AUC score
print(classification_report(y_test, y_pred))
print("ROC-AUC:", roc_auc_score(y_test, voting_model.predict_proba(X_test)[:, 1]))

# Save model and scaler
pickle.dump(voting_model, open("voting_model.pkl", "wb"))
pickle.dump(scaler, open("scaler.pkl", "wb"))

# Optionally, save the accuracy to a text file for later use
with open("accuracy.txt", "w") as f:
    f.write(str(accuracy))
