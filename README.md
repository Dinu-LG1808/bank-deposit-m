Bank Term-Deposit Prediction: Classification Model Comparison
a. Problem Statement
Banks run telemarketing campaigns to sell term deposits, and calling every customer is costly. The goal of this project is to predict whether a customer will subscribe to a term deposit (`deposit` = yes/no) from their demographic, financial and campaign-contact details. Five classification models are trained and compared on the same dataset, and the saved models are served through an interactive Streamlit app.
b. Dataset Description
Source: Bank Marketing dataset (Kaggle / UCI), file `bank.csv`
Task: Binary classification
Target column: `deposit` (yes / no)
Total features: 16 input features + 1 target (17 columns), which meets the minimum of 12 features
Instances: about 11,000+ rows, which meets the minimum of 500
Test set: 2,233 rows x 17 columns (`test\\\_data.csv`)
Feature types: numeric (age, balance, day, duration, campaign, pdays, previous) and categorical (job, marital, education, default, housing, loan, contact, month, poutcome)
Preprocessing: scaling of numeric features and encoding of categorical features, saved inside each sklearn pipeline
c. GitHub Repository Link
<PASTE YOUR GITHUB REPO LINK HERE>
Live Streamlit App: <PASTE YOUR STREAMLIT LINK HERE>
d. Models Used
Comparison Table
ML Model Name	Accuracy	AUC	Precision	Recall	F1	MCC
Logistic Regression	0.8262	0.9071	0.8278	0.7996	0.8135	0.6513
Decision Tree	0.8168	0.8590	0.7814	0.8516	0.8150	0.6366
KNN	0.8173	0.8796	0.8199	0.7873	0.8033	0.6333
Naive Bayes	0.7201	0.8042	0.7837	0.5652	0.6568	0.4472
Support Vector Machine	0.8554	0.9194	0.8249	0.8819	0.8524	0.7124
Observations
ML Model Name	Observation about model performance
Logistic Regression	A strong, stable baseline. It has the second-best AUC (0.9071) and the highest precision (0.8278), with precision and recall well balanced. This suggests the classes are reasonably separable by a mostly linear boundary.
Decision Tree	Has the second-highest recall (0.8516) but the lowest precision among the non-Naive-Bayes models (0.7814), so it produces more false positives. Its AUC (0.8590) is lower than Logistic Regression, SVM and KNN, which is typical of a single tree that overfits and gives coarse probability estimates.
KNN	Performs similarly to the Decision Tree on accuracy (0.8173) but with better AUC (0.8796) and precision (0.8199). It relies on distance, so it benefits from the feature scaling in the pipeline, but it is slower at prediction time and weaker than SVM and Logistic Regression.
Naive Bayes	The weakest model on every metric except precision. Recall is only 0.5652, so it misses many real subscribers, and MCC is 0.4472. The independence assumption between features likely does not hold here, since features such as balance, housing, loan and campaign details are correlated.
Support Vector Machine	The best model overall. It leads on accuracy (0.8554), AUC (0.9194), recall (0.8819), F1 (0.8524) and MCC (0.7124), while keeping precision (0.8249) close to the best. The kernel captures non-linear patterns that Logistic Regression cannot.
Overall Winner	Support Vector Machine: highest Accuracy, AUC, Recall, F1 and MCC, with the most balanced and robust performance on this dataset.
Repository Structure
```
project-folder/
|-- app.py
|-- requirements.txt
|-- README.md
|-- test\\\_data.csv
|-- ML\\\_Assignment\\\_2.ipynb
|-- model/
    |-- logistic\\\_regression.joblib
    |-- decision\\\_tree.joblib
    |-- knn.joblib
    |-- naive\\\_bayes.joblib
    |-- svm.joblib
```
How to Run Locally
```
pip install -r requirements.txt
streamlit run app.py
```
Streamlit App Features
Upload a test CSV (must contain all input features and the `deposit` column)
Dropdown to select one of the 5 saved models
Display of Accuracy, AUC, Precision, Recall, F1 and MCC
Confusion matrix and classification report
One-click comparison of all 5 models on the uploaded data
Saved models are loaded directly; no retraining happens in the app
