# diabetes-readmission-prediction
Machine learning model to predict 30-day readmission risk for diabetic patients
This project focuses on building a machine learning pipeline to predict 30-day hospital readmission risk among diabetic patients. The work is designed with real-world healthcare constraints in mind, particularly class imbalance, model stability, and business-aligned evaluation metrics.

Problem Statement
Unplanned hospital readmissions significantly increase healthcare costs and indicate suboptimal patient outcomes. The goal of this project is to identify patients who are at high risk of being readmitted within 30 days of discharge so that timely interventions can be prioritized.

Key Work Done
The project implements an end-to-end machine learning workflow. This includes data preprocessing with appropriate handling of missing values and feature transformations, class imbalance handling using SMOTE within a leakage-safe pipeline, comparison of multiple classification algorithms, and stratified cross-validation using ROC-AUC as the primary performance metric. A custom probability threshold is applied to improve recall, aligning the model with healthcare risk-management objectives. Feature importance analysis is also performed to improve interpretability.

Final Model
Random Forest was selected as the final model due to its consistent performance, robustness to noisy healthcare data, and interpretability. Key hyperparameters such as tree depth, minimum samples per leaf, and number of estimators were tuned to balance bias and variance. Class weighting and threshold optimization were used to further address the imbalanced nature of the problem.

Evaluation
Model performance was evaluated using ROC-AUC, precision, recall, F1-score, and confusion matrix analysis. ROC curves show stable generalization across validation folds. Threshold optimization significantly improved recall for high-risk patients without severely compromising overall model discrimination.

Business Relevance
The model can be integrated into hospital discharge workflows to flag high-risk patients in real time. This enables better care coordination, targeted follow-ups, reduced readmission penalties, and improved patient outcomes. The approach is scalable and can be extended to other chronic disease readmission use cases.

Project Structure
The repository contains the main notebook for data processing and model training, a processed dataset, and a detailed PDF report explaining methodology, evaluation, and business impact.

Tools Used
Python, Pandas, NumPy, scikit-learn, imbalanced-learn, Matplotlib, and Seaborn.

Additional Notes
This project prioritizes practical, industry-oriented decision making over purely academic model optimization. Detailed explanations and rationale are provided in the accompanying PDF report.
