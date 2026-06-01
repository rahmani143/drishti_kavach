import os
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import joblib

# File paths for saving data and the trained model
CSV_PATH = "criminal_records.csv"
MODEL_PATH = "random_forest_risk_model.pkl"

def generate_mock_tabular_data(database_npz_path="face_database.npz"):
    """
    Generates a mock history dataset matching the exact names 
    present in your face database.
    """
    if not os.path.exists(database_npz_path):
        print(f"Error: {database_npz_path} not found. Run your face recognition script first.")
        return
        
    with np.load(database_npz_path) as data:
        registered_names = data.files

    print(f"Found {len(registered_names)} individuals in the face database.")
    
    np.random.seed(42)
    rows = []
    
    # Create realistic mock feature rows for each individual
    for name in registered_names:
        prior_arrests = int(np.random.choice([0, 1, 2, 5, 8], p=[0.4, 0.3, 0.15, 0.1, 0.05]))
        has_active_warrant = int(np.random.choice([0, 1], p=[0.85, 0.15]))
        weapon_charges_count = int(np.random.choice([0, 1, 3], p=[0.7, 0.2, 0.1])) if prior_arrests > 0 else 0
        months_since_last_incident = float(np.random.uniform(1.0, 60.0)) if prior_arrests > 0 else 999.0
        
        # Rule-based risk assignment to simulate patterns for the model to learn
        if has_active_warrant == 1 or weapon_charges_count > 0 or prior_arrests >= 5:
            is_dangerous = 1  # High risk
        else:
            is_dangerous = 0  # Low risk
            
        rows.append({
            "identity_name": name,
            "prior_arrests": prior_arrests,
            "has_active_warrant": has_active_warrant,
            "weapon_charges_count": weapon_charges_count,
            "months_since_last_incident": months_since_last_incident,
            "is_dangerous": is_dangerous
        })
        
    df = pd.DataFrame(rows)
    df.to_csv(CSV_PATH, index=False)
    print(f"Tabular history database generated and saved to {CSV_PATH}")

def train_risk_model():
    """Trains a Random Forest Classifier on the generated tabular histories."""
    if not os.path.exists(CSV_PATH):
        print(f"Error: {CSV_PATH} missing. Generate the dataset first.")
        return
        
    df = pd.read_csv(CSV_PATH)
    
    # Features used for mathematical prediction (ignoring the name column)
    X = df[["prior_arrests", "has_active_warrant", "weapon_charges_count", "months_since_last_incident"]]
    y = df["is_dangerous"]
    
    # Train the ensemble classifier
    print("Training Random Forest risk prediction model...")
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X, y)
    
    # Save the trained model parameters to disk
    joblib.dump(clf, MODEL_PATH)
    print(f"Model trained successfully and exported to {MODEL_PATH}")

def assess_risk_by_name(name):
    """Retrieves a person's tabular file and uses Random Forest to evaluate risk."""
    if not os.path.exists(MODEL_PATH) or not os.path.exists(CSV_PATH):
        print("Required model or database files are missing.")
        return
        
    df = pd.read_csv(CSV_PATH)
    person_record = df[df["identity_name"] == name]
    
    if person_record.empty:
        print(f"No background record exists for {name}. System defaults to safe/unknown profile.")
        return
        
    # Extract feature inputs
    features = person_record[["prior_arrests", "has_active_warrant", "weapon_charges_count", "months_since_last_incident"]]
    
    clf = joblib.load(MODEL_PATH)
    
    # Execute prediction
    prediction = clf.predict(features)[0]
    probabilities = clf.predict_proba(features)[0]
    risk_percentage = probabilities[1] * 100
    
    status = "HIGH RISK / THREAT DETECTED" if prediction == 1 else "LOW RISK / CLEAR"
    print(f"\n--- Criminal Background Report for {name} ---")
    print(f"Prior Arrests: {person_record['prior_arrests'].values[0]}")
    print(f"Active Warrants: {person_record['has_active_warrant'].values[0]}")
    print(f"Weapon Charges: {person_record['weapon_charges_count'].values[0]}")
    print(f"Evaluation: {status} ({risk_percentage:.1f}% Threat Probability)")

if __name__ == "__main__":
    # 1. Generate the CSV containing history variables linked to your face database names
    generate_mock_tabular_data()
    
    # 2. Train the Random Forest classifier on those variables
    train_risk_model()
    
    # 3. Test evaluation step using one of your registered names
    assess_risk_by_name("zeenat_aman")