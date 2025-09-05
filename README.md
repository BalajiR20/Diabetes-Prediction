# Diabetes Prediction using Machine Learning 🧠💉

This project focuses on building and evaluating various machine learning models to predict the likelihood of diabetes based on patient data. The models were trained and tested on a standard dataset and evaluated based on accuracy and consistency.

## 📌 Project Overview

The goal of this project is to compare multiple ML models and select the best-performing one for diabetes prediction. We have used popular classification algorithms and fine-tuned them to achieve optimal results.

## 🗂️ Dataset

The dataset contains various medical parameters such as:
- Pregnancies
- Glucose
- Blood Pressure
- Skin Thickness
- Insulin
- BMI
- Diabetes Pedigree Function
- Age
- Outcome (Target variable: 1 = diabetic, 0 = non-diabetic)

## 🧪 Models Used

- Logistic Regression  
- K-Nearest Neighbors (KNN)  
- Decision Tree  
- Random Forest  
- Support Vector Machine (SVM)  
- XGBoost (Gradient Boosting)  
- MLP 

## 🔍 Model Evaluation

All models were evaluated using accuracy and standard deviation via cross-validation.

| Model                  | Accuracy (%) | Std. Dev |
|------------------------|--------------|----------|
| Logistic Regression    | 84.48        | ±2.68    |
| K-Nearest Neighbors    | 87.81        | ±1.96    |
| Decision Tree          | 97.61        | ±1.55    |
| Random Forest          | 97.93        | ±1.12    |
| SVM                    | 95.54        | ±1.02    |
| XGBoost                | 95.29        | ±2.32    |
| MLP            | 97.04        | ±1.40    |

## 🛠️ Hyperparameter Tuning

To optimize model performance, **GridSearchCV** was used to search for the best hyperparameters by evaluating multiple combinations through cross-validation. This helped in selecting the optimal values for parameters like `n_estimators`, `max_depth`, and `learning_rate` across different models, improving accuracy and reducing overfitting.

## 📈 Conclusion

Random Forest achieved the best overall performance, with high accuracy and low standard deviation. MLP (ANN) and Decision Tree also showed strong results. GridSearchCV played a key role in optimizing the models.

## 💻 Tools & Libraries

- Python  
- scikit-learn  
- XGBoost  
- NumPy & Pandas  
- Matplotlib & Seaborn  
- Jupyter Notebook


