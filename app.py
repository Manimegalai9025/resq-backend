from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import numpy as np
import sqlite3
import hashlib
import os
from datetime import datetime

app = Flask(__name__)
CORS(app)

# Load models at startup
print("🔄 Loading ML models...")
try:
    with open('donor_eligibility_model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    with open('label_encoder.pkl', 'rb') as f:
        label_encoder = pickle.load(f)
    print("✅ Models loaded successfully!")
except Exception as e:
    print(f"❌ Error loading models: {e}")
    model = None
    scaler = None
    label_encoder = None

def init_db():
    """Initialize SQLite database with required tables"""
    conn = sqlite3.connect('resq.db')
    c = conn.cursor()
    
    # Users table
    c.execute('''CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_type TEXT NOT NULL,
        email TEXT UNIQUE NOT NULL,
        password_hash TEXT NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )''')
    
    # Donors table
    c.execute('''CREATE TABLE IF NOT EXISTS donors (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER,
        name TEXT, 
        age INTEGER, 
        gender TEXT, 
        weight REAL, 
        blood_group TEXT,
        last_donation_months INTEGER, 
        has_illness INTEGER, 
        hemoglobin REAL,
        systolic_bp INTEGER, 
        diastolic_bp INTEGER, 
        phone TEXT,
        latitude REAL, 
        longitude REAL,
        eligibility_status TEXT, 
        eligibility_code INTEGER,
        FOREIGN KEY (user_id) REFERENCES users(id)
    )''')
    
    # Blood requests table
    c.execute('''CREATE TABLE IF NOT EXISTS blood_requests (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        hospital_name TEXT, 
        blood_group TEXT, 
        units_required INTEGER,
        patient_condition TEXT, 
        severity_level TEXT,
        latitude REAL, 
        longitude REAL,
        contact_phone TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )''')
    
    conn.commit()
    conn.close()
    print("✅ Database initialized")

def hash_password(password):
    """Hash password using SHA256"""
    return hashlib.sha256(password.encode()).hexdigest()

def predict_eligibility(age, gender, weight, last_donation, has_illness, hemoglobin, systolic_bp, diastolic_bp):
    """Predict donor eligibility using ML model"""
    if model is None or scaler is None:
        return 'Unknown', 0, 0.0
    
    # Encode gender (Male = 1, Female = 0)
    gender_encoded = 1 if gender.lower() == 'male' else 0
    
    # Prepare features in training order
    features = np.array([[
        age, weight, last_donation, has_illness, 
        hemoglobin, systolic_bp, diastolic_bp, gender_encoded
    ]])
    
    # Scale and predict
    features_scaled = scaler.transform(features)
    prediction = model.predict(features_scaled)[0]
    probability = model.predict_proba(features_scaled)[0]
    
    # Map prediction to status
    labels = {
        1: 'Eligible', 
        0: 'Temporarily Not Eligible', 
        -1: 'Permanently Not Eligible'
    }
    
    return labels[prediction], int(prediction), float(max(probability))

# ==================== API ENDPOINTS ====================

@app.route('/')
def home():
    """Health check endpoint"""
    return jsonify({
        'status': 'online',
        'message': 'RESQ Blood Donor API',
        'version': '1.0',
        'models_loaded': model is not None,
        'endpoints': {
            'register_donor': '/api/register_donor (POST)',
            'login': '/api/login (POST)',
            'predict': '/predict (POST)',
            'get_requests': '/api/get_requests (GET)',
            'create_request': '/api/create_request (POST)'
        }
    })

@app.route('/predict', methods=['POST'])
def predict():
    """
    Predict donor eligibility - for Android app
    Expected JSON: {age, gender, weight, lastDonationMonths, hasIllness, 
                   hemoglobin, systolicBp, diastolicBp}
    """
    try:
        if model is None or scaler is None:
            return jsonify({
                'error': 'Models not loaded',
                'message': 'Server error - ML models unavailable'
            }), 500
        
        data = request.json
        
        # Validate required fields
        required = ['age', 'gender', 'weight', 'lastDonationMonths', 
                   'hemoglobin', 'systolicBp', 'diastolicBp']
        for field in required:
            if field not in data:
                return jsonify({
                    'error': f'Missing field: {field}',
                    'message': 'Invalid request data'
                }), 400
        
        # Extract features
        age = float(data['age'])
        gender = data['gender']
        weight = float(data['weight'])
        last_donation = float(data['lastDonationMonths'])
        has_illness = int(data.get('hasIllness', 0))
        hemoglobin = float(data['hemoglobin'])
        systolic_bp = float(data['systolicBp'])
        diastolic_bp = float(data['diastolicBp'])
        
        # Get prediction
        eligibility_status, eligibility_code, confidence = predict_eligibility(
            age, gender, weight, last_donation, has_illness,
            hemoglobin, systolic_bp, diastolic_bp
        )
        
        return jsonify({
            'eligibility': eligibility_status,
            'eligibilityCode': eligibility_code,
            'confidence': round(confidence * 100, 2),
            'message': get_message(eligibility_code, age, weight, hemoglobin, 
                                  last_donation, systolic_bp, diastolic_bp, gender)
        }), 200
        
    except Exception as e:
        print(f"❌ Prediction error: {str(e)}")
        return jsonify({
            'error': str(e),
            'message': 'Prediction failed'
        }), 400

def get_message(code, age, weight, hemoglobin, last_donation, systolic_bp, diastolic_bp, gender):
    """Generate detailed eligibility message"""
    if code == 1:
        return "✅ You are eligible to donate blood! Thank you for saving lives."
    
    elif code == 0:
        reasons = []
        if last_donation < 3:
            reasons.append(f"recent donation ({last_donation} months ago)")
        if weight < 50:
            reasons.append(f"low weight ({weight}kg)")
        if gender.lower() == 'male' and hemoglobin < 13.0:
            reasons.append(f"low hemoglobin ({hemoglobin}g/dL)")
        elif gender.lower() == 'female' and hemoglobin < 12.5:
            reasons.append(f"low hemoglobin ({hemoglobin}g/dL)")
        if systolic_bp > 140 or systolic_bp < 100 or diastolic_bp > 90 or diastolic_bp < 60:
            reasons.append("blood pressure out of range")
        
        if reasons:
            return f"⏳ Temporarily not eligible: {', '.join(reasons)}"
        return "⏳ Temporarily not eligible. Please try again later."
    
    else:  # code == -1
        if age < 18:
            return "❌ You must be 18 years or older to donate blood."
        elif age > 65:
            return "❌ Donor age limit exceeded (maximum 65 years)."
        return "❌ Permanently not eligible. Please consult medical professionals."

@app.route('/api/register_donor', methods=['POST'])
def register_donor():
    """Register new donor with eligibility check"""
    try:
        data = request.json
        conn = sqlite3.connect('resq.db')
        c = conn.cursor()
        
        # Check if email exists
        c.execute('SELECT id FROM users WHERE email = ?', (data['email'],))
        if c.fetchone():
            conn.close()
            return jsonify({
                'success': False, 
                'message': 'Email already registered'
            }), 400
        
        # Predict eligibility
        eligibility_status, eligibility_code, confidence = predict_eligibility(
            data['age'], data['gender'], data['weight'], 
            data['last_donation_months'], data['has_illness'],
            data['hemoglobin'], data['systolic_bp'], data['diastolic_bp']
        )
        
        # Create user account
        password_hash = hash_password(data['password'])
        c.execute('INSERT INTO users (user_type, email, password_hash) VALUES (?, ?, ?)',
                  ('donor', data['email'], password_hash))
        user_id = c.lastrowid
        
        # Create donor profile
        c.execute('''INSERT INTO donors (
            user_id, name, age, gender, weight, blood_group,
            last_donation_months, has_illness, hemoglobin, 
            systolic_bp, diastolic_bp, phone, latitude, longitude,
            eligibility_status, eligibility_code
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''',
            (user_id, data['name'], data['age'], data['gender'], 
             data['weight'], data['blood_group'], data['last_donation_months'],
             data['has_illness'], data['hemoglobin'], data['systolic_bp'],
             data['diastolic_bp'], data['phone'], data.get('latitude'),
             data.get('longitude'), eligibility_status, eligibility_code))
        
        donor_id = c.lastrowid
        conn.commit()
        conn.close()
        
        return jsonify({
            'success': True,
            'donor_id': donor_id,
            'user_id': user_id,
            'eligibility_status': eligibility_status,
            'eligibility_code': eligibility_code,
            'confidence': round(confidence * 100, 2)
        }), 201
        
    except Exception as e:
        print(f"❌ Registration error: {str(e)}")
        return jsonify({
            'success': False, 
            'message': str(e)
        }), 500

@app.route('/api/login', methods=['POST'])
def login():
    """User login endpoint"""
    try:
        data = request.json
        password_hash = hash_password(data['password'])
        
        conn = sqlite3.connect('resq.db')
        conn.row_factory = sqlite3.Row
        c = conn.cursor()
        
        # Verify credentials
        c.execute('SELECT id, user_type FROM users WHERE email = ? AND password_hash = ?',
                  (data['email'], password_hash))
        user = c.fetchone()
        
        if not user:
            conn.close()
            return jsonify({
                'success': False, 
                'message': 'Invalid email or password'
            }), 401
        
        user_id = user['id']
        user_type = user['user_type']
        
        # Get donor profile if user is a donor
        profile = {}
        if user_type == 'donor':
            c.execute('SELECT * FROM donors WHERE user_id = ?', (user_id,))
            donor_row = c.fetchone()
            if donor_row:
                profile = dict(donor_row)
        
        conn.close()
        
        return jsonify({
            'success': True,
            'user_id': user_id,
            'user_type': user_type,
            'profile': profile
        }), 200
        
    except Exception as e:
        print(f"❌ Login error: {str(e)}")
        return jsonify({
            'success': False, 
            'message': str(e)
        }), 500

@app.route('/api/get_requests', methods=['GET'])
def get_requests():
    """Get all blood requests"""
    try:
        conn = sqlite3.connect('resq.db')
        conn.row_factory = sqlite3.Row
        c = conn.cursor()
        
        c.execute('SELECT * FROM blood_requests ORDER BY created_at DESC LIMIT 50')
        requests = [dict(row) for row in c.fetchall()]
        
        conn.close()
        return jsonify({
            'success': True, 
            'requests': requests
        }), 200
        
    except Exception as e:
        print(f"❌ Get requests error: {str(e)}")
        return jsonify({
            'success': False, 
            'message': str(e)
        }), 500

@app.route('/api/create_request', methods=['POST'])
def create_request():
    """Create new blood request"""
    try:
        data = request.json
        conn = sqlite3.connect('resq.db')
        c = conn.cursor()
        
        c.execute('''INSERT INTO blood_requests (
            hospital_name, blood_group, units_required,
            patient_condition, severity_level, latitude, longitude, contact_phone
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)''',
            (data.get('hospital_name'), data.get('blood_group'),
             data.get('units_required'), data.get('patient_condition'),
             data.get('severity_level'), data.get('latitude'),
             data.get('longitude'), data.get('contact_phone')))
        
        request_id = c.lastrowid
        conn.commit()
        conn.close()
        
        return jsonify({
            'success': True,
            'request_id': request_id,
            'message': 'Blood request created successfully'
        }), 201
        
    except Exception as e:
        print(f"❌ Create request error: {str(e)}")
        return jsonify({
            'success': False, 
            'message': str(e)
        }), 500

if __name__ == '__main__':
    init_db()
    port = int(os.environ.get('PORT', 5000))
    print("\n" + "="*60)
    print("🚀 RESQ Backend Server Started")
    print("="*60)
    print(f"📍 Running on: http://0.0.0.0:{port}")
    print("✅ Database initialized")
    print(f"✅ Models loaded: {model is not None}")
    print("="*60 + "\n")
    app.run(host='0.0.0.0', port=port)