import pandas as pd
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score, mean_squared_error
import numpy as np

# Define paths based on the requested structure
DATA_DIR = "dataset"
MODEL_DIR = "model"
DATA_PATH = os.path.join(DATA_DIR, "house_price_dataset.csv")
MODEL_PATH = os.path.join(MODEL_DIR, "gradient_boosting_model.pkl")

# Create directories if they don't exist
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

def train_and_save_model():
    """
    Loads the house price dataset, preprocesses it, trains a Gradient Boosting Regressor,
    and saves the trained pipeline.
    """
    # Load the dataset
    try:
        df = pd.read_csv(DATA_PATH)
    except FileNotFoundError:
        print(f"Dataset not found at {DATA_PATH}. Please make sure the file exists.")
        return

    # Separate features (X) and target (y)
    X = df.drop('Price', axis=1)
    y = df['Price']

    # Identify categorical and numerical columns
    categorical_features = X.select_dtypes(include=['object']).columns
    numerical_features = X.select_dtypes(include=['int64', 'float64']).columns

    # Create preprocessing pipelines for numerical and categorical data
    # Numerical data will be scaled
    numerical_transformer = Pipeline(steps=[
        ('scaler', StandardScaler())
    ])

    # Categorical data will be one-hot encoded
    categorical_transformer = Pipeline(steps=[
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    # Combine preprocessing pipelines using ColumnTransformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_features),
            ('cat', categorical_transformer, categorical_features)
        ],
        remainder='passthrough'
    )

    # Create the full pipeline with the preprocessor and the classifier
    # The GradientBoostingRegressor is the core of this model
    model_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42))
    ])

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train the model
    print("Training the Gradient Boosting Regressor model...")
    model_pipeline.fit(X_train, y_train)
    print("Training complete.")

    # Evaluate the model
    y_pred = model_pipeline.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)

    print(f"Model R-squared on the test set: {r2:.2f}")
    print(f"Model Mean Squared Error on the test set: {mse:.2f}")
    print(f"Model Root Mean Squared Error on the test set: {rmse:.2f}")

    # Save the trained pipeline
    print(f"Saving the trained model to {MODEL_PATH}...")
    joblib.dump(model_pipeline, MODEL_PATH)
    print("Model saved successfully.")

if __name__ == "__main__":
    train_and_save_model()
