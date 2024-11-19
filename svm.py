import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, classification_report
from tqdm import tqdm

try:
    # Load the dataset
    df = pd.read_csv('transformed.csv', encoding='ISO-8859-1')
    
    # Check if essential columns are present
    essential_columns = [
        'winner', 'match_duration_seconds',
        'radiant_player_1_kills', 'radiant_player_2_kills', 'radiant_player_3_kills', 
        'radiant_player_4_kills', 'radiant_player_5_kills', 'dire_player_1_kills', 
        'dire_player_2_kills', 'dire_player_3_kills', 'dire_player_4_kills', 
        'dire_player_5_kills', 'radiant_player_1_deaths', 'radiant_player_2_deaths', 
        'radiant_player_3_deaths', 'radiant_player_4_deaths', 'radiant_player_5_deaths', 
        'dire_player_1_deaths', 'dire_player_2_deaths', 'dire_player_3_deaths', 
        'dire_player_4_deaths', 'dire_player_5_deaths', 'radiant_player_1_assists', 
        'radiant_player_2_assists', 'radiant_player_3_assists', 'radiant_player_4_assists', 
        'radiant_player_5_assists', 'dire_player_1_assists', 'dire_player_2_assists', 
        'dire_player_3_assists', 'dire_player_4_assists', 'dire_player_5_assists'
    ]
    
    for col in essential_columns:
        if col not in df.columns:
            raise KeyError(f"Missing essential column in dataset: {col}")
 
    # Define initial features and feature groups for progressive addition
    base_features = essential_columns[1:]  # exclude 'winner' from features
    feature_groups = [
        [],  
        ['radiant_player_1_hero_id', 'radiant_player_2_hero_id', 'radiant_player_3_hero_id', 
        'radiant_player_4_hero_id', 'radiant_player_5_hero_id', 'dire_player_1_hero_id', 
        'dire_player_2_hero_id', 'dire_player_3_hero_id', 'dire_player_4_hero_id', 
        'dire_player_5_hero_id'],
        [f'{team}_player_{i}_networth' for team in ['radiant', 'dire'] for i in range(1, 6)],  
        [  
            'radiant_player_1_position', 'radiant_player_2_position', 'radiant_player_3_position', 'radiant_player_4_position', 'radiant_player_5_position', 
            'dire_player_1_position', 'dire_player_2_position', 'dire_player_3_position', 'dire_player_4_position', 'dire_player_5_position',
            'radiant_player_1_lane', 'radiant_player_2_lane', 'radiant_player_3_lane', 'radiant_player_4_lane', 'radiant_player_5_lane', 
            'dire_player_1_lane', 'dire_player_2_lane', 'dire_player_3_lane', 'dire_player_4_lane', 'dire_player_5_lane',
            'radiant_player_1_role', 'radiant_player_2_role', 'radiant_player_3_role', 'radiant_player_4_role', 'radiant_player_5_role', 
            'dire_player_1_role', 'dire_player_2_role', 'dire_player_3_role', 'dire_player_4_role', 'dire_player_5_role'
        ],
        ['first_blood_time_seconds', 'dire_kills', 'radiant_kills']  # Fourth iteration: Add first blood and kill-related features
    ]

    # Prepare the target
    y = df['winner']

    # Open file to save results
    with open('svm_results.txt', 'w') as f:
        f.write("Feature Addition Results\n")
        f.write("=========================\n\n")

        current_features = base_features[:]
        
        for idx, group in enumerate(tqdm(feature_groups, desc="Adding feature groups"), start=1):
            # For first iteration, use only base features
            if idx > 0:
                missing_features = [feature for feature in group if feature not in df.columns]
                if missing_features:
                    raise KeyError(f"Missing features in dataset: {missing_features}")
                current_features.extend(group)

            try:
                # Select current features
                X = df[current_features]

                # Split the data into training and testing sets (80-20 split)
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                
                # Standardize the features (not sure about this but will c later)
                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                X_test_scaled = scaler.transform(X_test)

                # SVM classifier with optimal C parameter (C=1)
                svm = SVC(C=1, kernel='linear', probability=True, random_state=42)
                svm.fit(X_train_scaled, y_train)

                # Predict on the test set
                y_pred = svm.predict(X_test_scaled)
                y_prob = svm.predict_proba(X_test_scaled)[:, 1]  # Probabilities for AUC calculation

                # Calculate metrics
                accuracy = accuracy_score(y_test, y_pred)
                f1 = f1_score(y_test, y_pred)
                precision = precision_score(y_test, y_pred)
                recall = recall_score(y_test, y_pred)
                auc = roc_auc_score(y_test, y_prob)

                # Save the results
                f.write(f"Features used in iteration {idx}: {current_features}\n")
                f.write(f"Accuracy: {accuracy:.4f}\n")
                f.write(f"AUC: {auc:.4f}\n")
                f.write(f"F1-Score: {f1:.4f}\n")
                f.write(f"Precision: {precision:.4f}\n")
                f.write(f"Recall: {recall:.4f}\n")
                f.write(f"Classification Report:\n{classification_report(y_test, y_pred)}\n")
                f.write("-" * 50 + "\n")

            except Exception as e:
                f.write(f"Error during iteration {idx}: {e}\n")
                f.write("-" * 50 + "\n")
    
    print("Process completed. Results are saved in 'svm_results.txt'.")

except FileNotFoundError:
    print("The dataset file 'transformed.csv' was not found. Please check the file path.")

except KeyError as e:
    print(f"Dataset is missing an essential column: {e}")

except Exception as e:
    print(f"An unexpected error occurred: {e}")
