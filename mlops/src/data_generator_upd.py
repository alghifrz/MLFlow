import pandas as pd
import numpy as np
import math

def calculate_psi(expected, actual, bins=10):
    """Calculate the PSI between two distributions"""
    discretizer = KBinsDiscretizer(n_bins=bins, encode='ordinal', strategy='uniform')
    expected_binned = discretizer.fit_transform(expected.reshape(-1, 1))
    actual_binned = discretizer.transform(actual.reshape(-1, 1))

    expected_freq = np.bincount(expected_binned.flatten(), minlength=bins) / len(expected)
    actual_freq = np.bincount(actual_binned.flatten(), minlength=bins) / len(actual)

    # Calculate PSI
    psi_value = np.sum((actual_freq - expected_freq) * np.log(actual_freq / expected_freq))
    return psi_value

def generate_data(n_samples, drift_params=None):
    """Generate synthetic customer churn data"""
    data = {
        'usage_mins': np.random.normal(600, 100, n_samples),
        'monthly_bill': np.random.normal(70, 20, n_samples),
        'support_calls': np.random.poisson(3, n_samples)
    }
    
    # Apply drift if specified
    if drift_params:
        for feature, (shift, scale) in drift_params.items():
            data[feature] = data[feature] * scale + shift
    
    df = pd.DataFrame(data)
    
    # Generate target (churn probability affected by features)
    churn_prob = 1 / (1 + np.exp(-(
        -5 + 
        0.005 * df['usage_mins'] + 
        0.02 * df['monthly_bill'] +
        0.2 * df['support_calls']
    )))
    df['churn'] = (np.random.random(n_samples) < churn_prob).astype(int)
    
    return df

def detect_drift(train_data, new_data):
    """Detect drift by calculating PSI for each feature"""
    features = ['usage_mins', 'monthly_bill', 'support_calls']
    drift_results = {}
    
    for feature in features:
        psi_value = calculate_psi(train_data[feature].values, new_data[feature].values)
        drift_results[feature] = psi_value
        print(f"Feature: {feature}, PSI: {psi_value}")
    
    return drift_results

# Generate datasets
if __name__ == "__main__":
    # Initial training data
    train_data = generate_data(1000)
    train_data.to_csv('data/train.csv', index=False)
    
    # New data with drift
    drift_params = {
        'usage_mins': (5, 1.5),    # Increase mean and variance
        'monthly_bill': (2, 1.6),   # Slight increase
        'support_calls': (0.5, 1.1)    # More variance in support calls
    }
    new_data = generate_data(1000, drift_params)
    new_data.to_csv('data/new_data.csv', index=False)

    # Detect drift
    drift_results = detect_drift(train_data, new_data)

    # Output drift status based on PSI
    for feature, psi in drift_results.items():
        if psi < 0.1:
            drift_status = "No drift"
        elif 0.1 <= psi < 0.2:
            drift_status = "Moderate drift"
        else:
            drift_status = "High drift"
        print(f"Feature: {feature}, Drift Status: {drift_status}")
