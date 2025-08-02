import pandas as pd
import numpy as np
import joblib
import warnings
warnings.filterwarnings('ignore')

def load_model_and_preprocessors():
    """Load the trained model and preprocessing objects"""
    try:
        # Load model
        model = joblib.load('best_model.pkl')
        print("✓ Model loaded successfully")
        
        # Load preprocessing objects
        num_imputer = joblib.load('num_imputer.pkl')
        encoders = joblib.load('encoders.pkl')
        scaler = joblib.load('scaler.pkl')
        y_inv_map = joblib.load('y_inv_map.pkl')
        
        print("✓ Preprocessing objects loaded successfully")
        
        return model, num_imputer, encoders, scaler, y_inv_map
        
    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
        print("Please ensure you have run the training pipeline first.")
        return None, None, None, None, None

def preprocess_new_data(data, num_imputer, encoders, scaler, num_cols, cat_cols):
    """Preprocess new data using saved preprocessing objects"""
    
    # Handle missing values
    if data.isnull().sum().sum() > 0:
        print("Handling missing values...")
        data[num_cols] = num_imputer.transform(data[num_cols])
        if len(cat_cols) > 0:
            cat_imputer = joblib.load('cat_imputer.pkl')
            data[cat_cols] = cat_imputer.transform(data[cat_cols])
    
    # Encode categorical variables
    if len(cat_cols) > 0:
        print("Encoding categorical variables...")
        for col in cat_cols:
            if col in encoders:
                if hasattr(encoders[col], 'transform'):  # TargetEncoder
                    data[col] = encoders[col].transform(data[col])
                else:  # OneHotEncoder
                    ohe_cols = [f"{col}_{cat}" for cat in encoders[col].categories_[0]]
                    ohe_data = pd.DataFrame(encoders[col].transform(data[[col]]), 
                                          columns=ohe_cols, index=data.index)
                    data = pd.concat([data.drop(col, axis=1), ohe_data], axis=1)
    
    # Scale numeric features
    print("Scaling numeric features...")
    data[num_cols] = scaler.transform(data[num_cols])
    
    return data

def predict_personality(data_path, output_path='predictions.csv'):
    """Make predictions on new data"""
    
    print("=" * 60)
    print("PERSONALITY PREDICTION")
    print("=" * 60)
    
    # Load model and preprocessors
    model, num_imputer, encoders, scaler, y_inv_map = load_model_and_preprocessors()
    
    if model is None:
        return
    
    # Load new data
    try:
        new_data = pd.read_csv(data_path)
        print(f"✓ Data loaded: {new_data.shape}")
    except FileNotFoundError:
        print(f"❌ Error: Could not find {data_path}")
        return
    
    # Identify column types
    num_cols = new_data.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = new_data.select_dtypes(include=['object']).columns.tolist()
    
    # Remove ID column if present
    if 'id' in new_data.columns:
        ids = new_data['id']
        new_data = new_data.drop('id', axis=1)
    else:
        ids = range(len(new_data))
    
    # Preprocess data
    processed_data = preprocess_new_data(new_data, num_imputer, encoders, scaler, num_cols, cat_cols)
    
    # Make predictions
    print("Making predictions...")
    predictions = model.predict(processed_data)
    
    # Decode predictions
    predictions_decoded = [y_inv_map.get(i, i) for i in predictions]
    
    # Create results DataFrame
    results = pd.DataFrame({
        'id': ids,
        'Personality': predictions_decoded
    })
    
    # Save predictions
    results.to_csv(output_path, index=False)
    print(f"✓ Predictions saved to {output_path}")
    
    # Display statistics
    print(f"\nPREDICTION STATISTICS")
    print("-" * 40)
    print(f"Total predictions: {len(predictions_decoded)}")
    print(f"Prediction distribution:")
    pred_dist = pd.Series(predictions_decoded).value_counts()
    for personality, count in pred_dist.items():
        print(f"  {personality}: {count} ({count/len(predictions_decoded)*100:.1f}%)")
    
    # Show first few predictions
    print(f"\nFIRST 10 PREDICTIONS")
    print("-" * 40)
    print(results.head(10))
    
    return results

def predict_single_sample(sample_data):
    """Make prediction on a single sample (dictionary format)"""
    
    print("=" * 60)
    print("SINGLE SAMPLE PREDICTION")
    print("=" * 60)
    
    # Load model and preprocessors
    model, num_imputer, encoders, scaler, y_inv_map = load_model_and_preprocessors()
    
    if model is None:
        return
    
    # Convert to DataFrame
    sample_df = pd.DataFrame([sample_data])
    
    # Identify column types
    num_cols = sample_df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = sample_df.select_dtypes(include=['object']).columns.tolist()
    
    # Preprocess data
    processed_data = preprocess_new_data(sample_df, num_imputer, encoders, scaler, num_cols, cat_cols)
    
    # Make prediction
    prediction = model.predict(processed_data)[0]
    prediction_decoded = y_inv_map.get(prediction, prediction)
    
    # Get prediction probabilities if available
    if hasattr(model, 'predict_proba'):
        probabilities = model.predict_proba(processed_data)[0]
        prob_dict = {y_inv_map.get(i, i): prob for i, prob in enumerate(probabilities)}
    else:
        prob_dict = None
    
    print(f"Prediction: {prediction_decoded}")
    if prob_dict:
        print("Probabilities:")
        for personality, prob in prob_dict.items():
            print(f"  {personality}: {prob:.3f}")
    
    return prediction_decoded, prob_dict

if __name__ == "__main__":
    # Example usage
    print("Personality Classification - Prediction Script")
    print("=" * 60)
    
    # Option 1: Predict on a CSV file
    # predict_personality('test.csv', 'my_predictions.csv')
    
    # Option 2: Predict on a single sample
    # sample = {
    #     'feature1': 1.5,
    #     'feature2': 2.3,
    #     # ... add all required features
    # }
    # prediction, probabilities = predict_single_sample(sample)
    
    print("\nTo use this script:")
    print("1. For batch predictions: predict_personality('your_data.csv')")
    print("2. For single sample: predict_single_sample(your_sample_dict)") 