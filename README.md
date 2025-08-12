# Gradient_Boosting - House Price Predictor

## Overview

This project implements a **Gradient Boosting Regressor** to predict the price of a house based on its features.

The model is trained on a custom dataset and deployed through a **Flask** web application, allowing users to input measurements and get an instant price prediction.

## Project Structure

```
DataScience/
│
├── GradientBoosting/
│   ├── data/
│   │   └── house_price_dataset.csv
│   ├── model/
│   │   └── gradient_boosting_model.pkl
│   ├── static/
│   │   └── style.css
│   ├── templates/
│   │   ├── index.html
│   │   └── result.html
│   ├── model.py
│   ├── app.py
│   └── requirements.txt
```

## Installation & Setup

1. **Clone the repository**

    ```
    git clone <your-repo-url>
    cd "DataScience/GradientBoosting"
    ```

2. **Create a virtual environment (recommended)**

    ```
    python -m venv venv
    source venv/bin/activate    # For Linux/Mac
    venv\Scripts\activate       # For Windows
    ```

3. **Install dependencies**

    ```
    pip install -r requirements.txt
    ```

## Dataset

The dataset used is a custom **House Price Dataset** with the following features:

* **SquareFootage** (numeric)
* **Bedrooms** (numeric)
* **Bathrooms** (numeric)
* **YearBuilt** (numeric)
* **Location** (categorical)
* **Price** (Target: The price of the house)

## Why Gradient Boosting?

**Gradient Boosting** is a powerful ensemble machine learning algorithm for both classification and regression tasks. It builds models sequentially, where each new model corrects the errors of the previous one. This iterative process allows it to create a highly accurate predictive model by progressively reducing the residual error. It's known for its high performance and ability to handle complex, non-linear relationships in data.

## How to Run

1. **Train the Model**

    ```
    python model.py
    ```

    This will create:

    * `gradient_boosting_model.pkl` (trained model)

2. **Run the Flask App**

    ```
    python app.py
    ```

    Visit `http://127.0.0.1:5000/` in your browser.

## Prediction Goal

The application predicts the price of a house, for example: `$450,000.00`.

## Tech Stack

* **Python** – Core programming language
* **Pandas & NumPy** – Data manipulation
* **Scikit-learn** – Machine learning model training
* **Flask** – Web framework for deployment
* **HTML/CSS** – Frontend UI design

## Future Scope

* **Model Comparison:** Implement and compare the Gradient Boosting model with other regression algorithms (e.g., Linear Regression, Random Forest) to see which performs best on the dataset.
* **Feature Engineering:** Add new features, such as `age_of_house`, or interaction terms to potentially improve model performance.
* **Deployment:** Deploy the Flask application to a cloud platform like Heroku or Render for public access.
* **User Interface:** Enhance the web application with a more interactive UI, providing visualizations of the data or a confidence score for each prediction.


## Screen Shots

**Home Page:**

<img width="1920" height="1080" alt="Screenshot (29)" src="https://github.com/user-attachments/assets/2751b6e6-443b-41c4-a16c-c045580be74a" />


**Result Page:**

<img width="1920" height="1080" alt="Screenshot (30)" src="https://github.com/user-attachments/assets/10dac498-8d32-460a-aa4f-70484b9be43b" />

