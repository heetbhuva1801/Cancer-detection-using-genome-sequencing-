import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer

def load_data(file_path):
    data = pd.read_csv(file_path)
    return data

def preprocess_data(data):
    # Drop any columns that are non-numeric or irrelevant (like IDs)
    data = data.select_dtypes(include=[float, int]).copy()

    # Handle categorical data by encoding
    if 'diagnosis' in data.columns:
        le = LabelEncoder()
        data['diagnosis'] = le.fit_transform(data['diagnosis'])

    # Impute missing values with the median
    imputer = SimpleImputer(strategy='median')
    data_imputed = imputer.fit_transform(data)

    # Standardize the numeric features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(data_imputed)
    return X_scaled
