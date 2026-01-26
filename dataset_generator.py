import pandas as pd
import numpy as np
import random

np.random.seed(42)
random.seed(42)

def determine_eligibility(age, gender, weight, last_donation, has_illness, hemoglobin, systolic_bp, diastolic_bp):
    if age < 18 or age > 65: return -1
    if has_illness and random.random() < 0.7: return -1
    if weight < 50: return 0
    if last_donation < 3: return 0
    if gender == 'Male' and hemoglobin < 13.0: return 0
    if gender == 'Female' and hemoglobin < 12.5: return 0
    if systolic_bp > 140 or systolic_bp < 100: return 0
    if diastolic_bp > 90 or diastolic_bp < 60: return 0
    return 1

data = []
blood_groups = ['A+', 'A-', 'B+', 'B-', 'AB+', 'AB-', 'O+', 'O-']
blood_dist = [0.30, 0.08, 0.25, 0.07, 0.10, 0.05, 0.12, 0.03]

for i in range(5000):
    age = np.random.randint(16, 70)
    gender = random.choice(['Male', 'Female'])
    weight = max(40, min(120, np.random.normal(70 if gender == 'Male' else 60, 12)))
    blood_group = np.random.choice(blood_groups, p=blood_dist)
    last_donation = np.random.choice([0, 1, 2, 3, 4, 6, 8, 12, 18, 24])
    has_illness = 1 if random.random() < 0.20 else 0
    hemoglobin = round(max(10, min(18, np.random.normal(14.5 if gender == 'Male' else 13.0, 1.2))), 1)
    systolic_bp = np.random.randint(100, 160)
    diastolic_bp = np.random.randint(60, 100)
    
    eligibility = determine_eligibility(age, gender, weight, last_donation, has_illness, hemoglobin, systolic_bp, diastolic_bp)
    
    data.append({
        'age': age, 'gender': gender, 'weight': round(weight, 1),
        'blood_group': blood_group, 'last_donation_months': last_donation,
        'has_illness': has_illness, 'hemoglobin': hemoglobin,
        'systolic_bp': systolic_bp, 'diastolic_bp': diastolic_bp,
        'eligibility': eligibility
    })

df = pd.DataFrame(data)
df.to_csv('donor_eligibility_dataset.csv', index=False)
print(f"✅ Dataset created: {len(df)} records")
print("\nEligibility Distribution:")
print(df['eligibility'].value_counts())