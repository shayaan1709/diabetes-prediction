# Diabetes Prediction

Machine learning project that predicts whether a patient has diabetes from medical measurements. It compares 6 classification models and shows which features matter most.

## Dataset
768 patients, 8 features (Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age) and a binary Outcome (1 = diabetes).

## Steps
1. Cleaned the data: replaced invalid zeros in Glucose, BloodPressure, SkinThickness, Insulin and BMI with the column median
2. Exploratory analysis: class balance, glucose by outcome, BMI distribution, correlation heatmap
3. 80/20 train-test split
4. Balanced the training set with SMOTE (400/214 became 400/400)
5. Trained Logistic Regression, KNN, Decision Tree, Random Forest, AdaBoost and Gradient Boosting
6. Compared them with accuracy, classification reports, confusion matrices and ROC curves
7. Checked feature importance with Random Forest

## Results (from the saved notebook run)

| Model | Accuracy |
|---|---|
| Logistic Regression | 79.9% |
| KNN | 89.0% |
| Decision Tree | 84.4% |
| Random Forest | 90.9% |
| AdaBoost | 92.2% |
| Gradient Boosting | 92.9% |

Ensemble models performed best. Glucose and BMI were the most important factors. The split has no fixed random seed, so numbers change slightly between runs.

## Tech
Python, pandas, NumPy, scikit-learn, imbalanced-learn, matplotlib, seaborn

## Run it
    pip install -r requirements.txt
    jupyter notebook diabetes.ipynb

Keep diabetes.csv in the same folder as the notebook.
