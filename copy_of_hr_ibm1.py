import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, accuracy_score
from imblearn.over_sampling import SMOTE
import xgboost as xgb
import pickle

# Load dataset
df = pd.read_csv('WA_Fn-UseC_-HR-Employee-Attrition.csv')

# Drop unused columns
df.drop(['EmployeeNumber', 'EmployeeCount', 'Over18', 'StandardHours'], axis=1, inplace=True)

# Encode target
df['Attrition'] = df['Attrition'].map({'Yes': 1, 'No': 0})

# Encode categorical features
label_encoders = {}
for col in df.select_dtypes(include='object'):
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# Selected features
features = [
    'MonthlyIncome', 'OverTime', 'Age', 'DailyRate', 'TotalWorkingYears',
    'MonthlyRate', 'HourlyRate', 'DistanceFromHome', 'YearsAtCompany',
    'NumCompaniesWorked', 'PercentSalaryHike', 'JobRole',
    'JobSatisfaction', 'EnvironmentSatisfaction', 'WorkLifeBalance',
    'JobInvolvement', 'BusinessTravel', 'Department', 'EducationField',
    'Education', 'StockOptionLevel', 'TrainingTimesLastYear'
]

# Define categorical columns (to exclude from SMOTE)
categorical_cols = ['OverTime', 'JobRole', 'BusinessTravel', 'Department', 'EducationField']

# Feature matrix and target
X = df[features]
y = df['Attrition']

# Split into train-test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# Model before SMOTE (for comparison)
model_before = xgb.XGBClassifier(
    n_estimators=200,
    learning_rate=0.1,
    max_depth=5,
    use_label_encoder=False,
    eval_metric='logloss',
    random_state=42
)
model_before.fit(X_train, y_train)
y_pred_before = (model_before.predict_proba(X_test)[:, 1] >= 0.30).astype(int)
print("📉 BEFORE SMOTE:")
print("Accuracy:", round(accuracy_score(y_test, y_pred_before) * 100, 2), "%")
print("Classification Report:\n", classification_report(y_test, y_pred_before))
print("ROC AUC Score:", roc_auc_score(y_test, model_before.predict_proba(X_test)[:, 1]))

# Apply SMOTE on numeric-only features
X_train_smote = X_train.drop(columns=categorical_cols)
smote = SMOTE(random_state=42)
X_train_bal_numeric, y_train_bal = smote.fit_resample(X_train_smote, y_train)

# Join back categorical features (from original X_train)
X_train_categoricals = X_train[categorical_cols].reset_index(drop=True)
X_train_bal = pd.concat([X_train_bal_numeric.reset_index(drop=True), X_train_categoricals.loc[X_train_bal_numeric.index]], axis=1)

print(f"\n✅ Applied SMOTE. Balanced classes: {y_train_bal.value_counts().to_dict()}")

# Train final model with SMOTE-applied data (all features)
model = xgb.XGBClassifier(
    n_estimators=250,
    learning_rate=0.08,
    max_depth=6,
    use_label_encoder=False,
    eval_metric='logloss',
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)
model.fit(X_train_bal, y_train_bal)

# Predict using custom threshold for better recall
y_proba = model.predict_proba(X_test)[:, 1]
threshold = 0.30
y_pred = (y_proba >= threshold).astype(int)

# Evaluate model
print("\n📈 AFTER SMOTE (Final Model):")
print("🔎 Accuracy:", round(accuracy_score(y_test, y_pred) * 100, 2), "%")
print("🔎 Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("🔎 Classification Report:\n", classification_report(y_test, y_pred))
print("🔎 ROC AUC Score:", roc_auc_score(y_test, y_proba))

# Feature importance
importance_df = pd.DataFrame({
    'Feature': X.columns,
    'Importance': model.feature_importances_
}).sort_values(by='Importance', ascending=False)

# Plot top features
plt.figure(figsize=(12, 6))
sns.barplot(data=importance_df.head(15), x='Importance', y='Feature', palette='viridis')
plt.title('Top 15 Features Influencing Attrition')
plt.tight_layout()
plt.show()

# Save model
with open('xgb_attrition_model_v2.pkl', 'wb') as f:
    pickle.dump(model, f)

print("Final model (SMOTE-trained) saved as 'xgb_attrition_model_v2.pkl'")
