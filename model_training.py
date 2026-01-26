
import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score, 
    precision_score, 
    recall_score, 
    f1_score,
    confusion_matrix, 
    classification_report
)

print("="*60)
print("RESQ - MODEL TRAINING & ACCURACY EVALUATION")
print("="*60)

# Load dataset
print("\n[1/6] Loading dataset...")
df = pd.read_csv('donor_eligibility_dataset.csv')
print(f"✅ Dataset loaded: {df.shape[0]} samples, {df.shape[1]} features")

# Show class distribution
print(f"\n[2/6] Class Distribution:")
print(df['eligibility'].value_counts())
print(f"\nClass Balance:")
for label, count in df['eligibility'].value_counts().items():
    percentage = (count / len(df)) * 100
    label_name = {1: 'Eligible', 0: 'Temporary', -1: 'Permanent'}[label]
    print(f"  {label_name:15s}: {count:4d} ({percentage:.1f}%)")

# Prepare features
print(f"\n[3/6] Preparing features...")
X = df[['age', 'weight', 'last_donation_months', 'has_illness', 
        'hemoglobin', 'systolic_bp', 'diastolic_bp', 'gender']]
y = df['eligibility']

# Encode gender
le_gender = LabelEncoder()
X['gender'] = le_gender.fit_transform(X['gender'])

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.3, random_state=42, stratify=y
)
print(f"✅ Training set: {X_train.shape[0]} samples")
print(f"✅ Test set: {X_test.shape[0]} samples")

# Train multiple models for comparison
print(f"\n[4/6] Training models...")

models = {}

# Model 1: Random Forest
print("\n  🌲 Random Forest Classifier...")
rf_model = RandomForestClassifier(
    n_estimators=100, 
    max_depth=10, 
    min_samples_split=5,
    random_state=42, 
    n_jobs=-1
)
rf_model.fit(X_train, y_train)
rf_pred = rf_model.predict(X_test)
models['Random Forest'] = {
    'model': rf_model,
    'predictions': rf_pred,
    'accuracy': accuracy_score(y_test, rf_pred)
}
print(f"     Accuracy: {models['Random Forest']['accuracy']:.4f}")

# Model 2: Logistic Regression
print("\n  📊 Logistic Regression...")
lr_model = LogisticRegression(max_iter=1000, random_state=42)
lr_model.fit(X_train, y_train)
lr_pred = lr_model.predict(X_test)
models['Logistic Regression'] = {
    'model': lr_model,
    'predictions': lr_pred,
    'accuracy': accuracy_score(y_test, lr_pred)
}
print(f"     Accuracy: {models['Logistic Regression']['accuracy']:.4f}")

# Model 3: Decision Tree
print("\n  🌳 Decision Tree Classifier...")
dt_model = DecisionTreeClassifier(max_depth=8, random_state=42)
dt_model.fit(X_train, y_train)
dt_pred = dt_model.predict(X_test)
models['Decision Tree'] = {
    'model': dt_model,
    'predictions': dt_pred,
    'accuracy': accuracy_score(y_test, dt_pred)
}
print(f"     Accuracy: {models['Decision Tree']['accuracy']:.4f}")

# Find best model
best_model_name = max(models, key=lambda x: models[x]['accuracy'])
best_model = models[best_model_name]['model']
best_predictions = models[best_model_name]['predictions']

print(f"\n[5/6] MODEL COMPARISON")
print("="*60)
print(f"{'Model':<25} {'Accuracy':<12} {'Status'}")
print("-"*60)
for name, data in sorted(models.items(), key=lambda x: x[1]['accuracy'], reverse=True):
    status = "🏆 BEST" if name == best_model_name else ""
    print(f"{name:<25} {data['accuracy']:<12.4f} {status}")
print("="*60)

# Detailed metrics for best model
print(f"\n[6/6] DETAILED METRICS - {best_model_name}")
print("="*60)

# Accuracy metrics
accuracy = accuracy_score(y_test, best_predictions)
precision = precision_score(y_test, best_predictions, average='weighted')
recall = recall_score(y_test, best_predictions, average='weighted')
f1 = f1_score(y_test, best_predictions, average='weighted')

print(f"\n📊 Overall Performance Metrics:")
print(f"   Accuracy  : {accuracy:.4f} ({accuracy*100:.2f}%)")
print(f"   Precision : {precision:.4f} ({precision*100:.2f}%)")
print(f"   Recall    : {recall:.4f} ({recall*100:.2f}%)")
print(f"   F1-Score  : {f1:.4f} ({f1*100:.2f}%)")

# Confusion Matrix
print(f"\n📈 Confusion Matrix:")
cm = confusion_matrix(y_test, best_predictions)
labels = ['Permanent (-1)', 'Temporary (0)', 'Eligible (1)']
print(f"\n{'':>18}" + "  ".join([f"{l:^18}" for l in labels]))
print("-" * 72)
for i, label in enumerate(labels):
    row = cm[i]
    print(f"{label:>18}  " + "  ".join([f"{val:^18}" for val in row]))

# Per-class metrics
print(f"\n📋 Per-Class Performance:")
print(classification_report(y_test, best_predictions, 
                          target_names=labels,
                          digits=4))

# Cross-validation
print(f"\n🔄 Cross-Validation (5-Fold):")
cv_scores = cross_val_score(best_model, X_scaled, y, cv=5)
print(f"   Fold Scores: {[f'{score:.4f}' for score in cv_scores]}")
print(f"   Mean Accuracy: {cv_scores.mean():.4f} (±{cv_scores.std():.4f})")

# Feature Importance (for Random Forest)
if best_model_name == 'Random Forest':
    print(f"\n🎯 Feature Importance:")
    feature_names = ['Age', 'Weight', 'Last Donation', 'Has Illness', 
                    'Hemoglobin', 'Systolic BP', 'Diastolic BP', 'Gender']
    importances = best_model.feature_importances_
    
    for name, importance in sorted(zip(feature_names, importances), 
                                   key=lambda x: x[1], reverse=True):
        bar = "█" * int(importance * 50)
        print(f"   {name:15s}: {importance:.4f} {bar}")

# Save best model
print(f"\n💾 Saving models...")
with open('donor_eligibility_model.pkl', 'wb') as f:
    pickle.dump(best_model, f)
with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)
with open('label_encoder.pkl', 'wb') as f:
    pickle.dump(le_gender, f)

print(f"✅ Models saved:")
print(f"   - donor_eligibility_model.pkl")
print(f"   - scaler.pkl")
print(f"   - label_encoder.pkl")

# Test predictions
print(f"\n🧪 TEST PREDICTIONS")
print("="*60)

def predict_eligibility(age, gender, weight, last_donation, has_illness, 
                       hemoglobin, systolic_bp, diastolic_bp):
    gender_encoded = 1 if gender.lower() == 'male' else 0
    features = np.array([[age, weight, last_donation, has_illness, 
                         hemoglobin, systolic_bp, diastolic_bp, gender_encoded]])
    features_scaled = scaler.transform(features)
    prediction = best_model.predict(features_scaled)[0]
    probability = best_model.predict_proba(features_scaled)[0]
    labels_map = {1: 'Eligible', 0: 'Temporarily Not Eligible', -1: 'Permanently Not Eligible'}
    return labels_map[prediction], prediction, max(probability)

test_cases = [
    (25, 'Male', 70, 6, 0, 14.5, 120, 80, "✅ Healthy Male Donor"),
    (30, 'Female', 55, 2, 0, 13.0, 115, 75, "⏳ Recent Donor (Temporary)"),
    (22, 'Female', 45, 12, 0, 12.8, 110, 70, "⚠️ Underweight Donor"),
    (68, 'Male', 75, 24, 0, 14.0, 125, 82, "❌ Elderly (Permanent)"),
    (19, 'Male', 65, 4, 0, 13.5, 118, 76, "✅ Young Eligible Donor"),
]

for age, gender, weight, last, ill, hb, sys, dia, desc in test_cases:
    result, code, conf = predict_eligibility(age, gender, weight, last, ill, hb, sys, dia)
    print(f"{desc:35s} → {result:30s} ({conf*100:.1f}% confidence)")

print("\n" + "="*60)
print("✅ MODEL TRAINING COMPLETE!")
print(f"🏆 Best Model: {best_model_name}")
print(f"📊 Final Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
print("="*60)
