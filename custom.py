import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier
import io
import joblib

# def read_input_data(data):
#     ###Commenting 11 and uncommenting 12 and 13 pushes the error to the model hook...something about binary data
#     data = pd.read_csv(data)
#     #data_str = io.StringIO(input_binary_data)
#     #data = pd.read_csv(data_str)
#     return data

def load_model(code_dir):
    return 1

def load_seg_model(seg):
    if seg == 1:
        with open("model1.pkl", "rb") as f:
            model = joblib.load(f)
    else:
        with open("model2.pkl", "rb") as f:
            model = joblib.load(f)
    return model

def transform(data, model):
    data = data.fillna(0)
    data['funded_amnt'] = np.where(data['funded_amnt'] > 0, np.log(data['funded_amnt']), 0)
    data['int_rate'] = data['int_rate'].str.rstrip('%').astype(float)
    data['derivedVar'] = data['funded_amnt'] + data['int_rate']
    # Apply Segmentation Logic
    data['row_id'] = range(1, len(data) + 1)
    data['seg'] = np.where(data['dti'] < 13, 1, 2)
    return data

def score(data, model, **kwargs):
    # Extract unique segment values from data
    segs = data['seg'].unique().tolist()

    # Initialize an empty dataframe to store the scored data
    scored = pd.DataFrame()

    # Loop through each segment
    for s in segs:
        # Filter data for the current segment
        seg_data = data[data['seg'] == s]

        # Load segment-specific model
        s_model = load_seg_model(s)

        # Make predictions using the segment-specific model
        pred = pd.DataFrame(s_model.predict_proba(seg_data.drop(['is_bad','seg','row_id'],axis=1)), columns=["0", "1"])

        # Combine predictions with segment-specific data
        seg_scored = pd.concat([seg_data['row_id'].reset_index(drop=True), pred], axis=1)
        seg_scored['seg'] = s

        # Append segment-scored data to the main scored dataframe
        scored = scored.append(seg_scored, ignore_index=True)

    # Join the scored data with the original data by 'row_id'
    scored = pd.merge(data, scored, on='row_id', how='left')

    # Reorder columns and select only the relevant columns
    scored = scored[['0', '1']]

    return scored
