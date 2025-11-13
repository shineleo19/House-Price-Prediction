# 🏠 House Price Prediction

This project builds a **Linear Regression model** to predict house prices using features such as square footage, number of bedrooms, bathrooms, and more.

---

## 📊 Overview

The goal of this project is to understand the relationship between house features and their selling prices, and to predict the price of a house based on its characteristics.

The dataset includes features like:
- Bedrooms  
- Bathrooms  
- Living area (sqft)  
- Lot area (sqft)  
- Year built  
- Location details (city, zip, state)

---

## ⚙️ Technologies Used

- **Python 3**
- **Pandas** – data analysis  
- **NumPy** – numerical computations  
- **Scikit-learn** – model training and evaluation  
- **Matplotlib / Seaborn** – data visualization  

---

## 🚀 How It Works

1. Load the dataset (e.g., from `data.csv`)
2. Perform data cleaning and preprocessing
3. Split the dataset into training and testing sets
4. Train a **Linear Regression model**
5. Evaluate model performance using:
   - Mean Squared Error (MSE)
   - R² Score

---

## 🧠 Model Workflow

```plaintext
Data → Preprocessing → Train/Test Split → Model Training → Evaluation → Predictions

## 🚀 How to Run

### 1️⃣ Clone the repository
```bash
1.git clone <repo url>
2.cd house-price-prediction
3.pip install -r requirements.txt
4.python main.py
