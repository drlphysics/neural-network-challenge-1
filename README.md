# neural-network-challenge-1
## Overview
This project involves using a neural network model to predict student loan repayment success based on various financial and personal characteristics of students. The analysis is divided into four main parts: preparing the data, compiling and evaluating a neural network model, making predictions, and discussing the creation of a recommendation system for student loans. The ultimate goal is to create a predictive model that assesses student loan repayment success and explore how to build a recommendation system for student loans.

## Repository Structure
student-loan-prediction/
│
├── data/
│   ├── student-loans.csv
│
├── notebooks/
│   ├── neural_network_model.ipynb
│
├── models/
│   ├── student_loans.keras
│
├── README.md
│
└── results/
    ├── classification_report.txt

## Instructions
### Part 1: Prepare the Data for Use in a Neural Network Model
Objective: Preprocess the data to be used in a neural network model.

Load Data: Read the data from the provided URL into a Pandas DataFrame and review the data to identify features and target variables.
Define Features and Target:
Features (X): All columns except the target column.
Target (y): The credit_ranking column.
Split Data: Split the features and target data into training and testing datasets using train_test_split.
Scale Data: Use StandardScaler from scikit-learn to scale the features data.
Output: Scaled training and testing datasets.

### Part 2: Compile and Evaluate a Model Using a Neural Network
Objective: Create, compile, and evaluate a deep neural network model using TensorFlow and Keras.

Design the Neural Network Model:
Define the number of input features based on the dataset.
Create a deep neural network with at least two layers, using the relu activation function for both layers.
Compile the Model: Use the binary_crossentropy loss function, adam optimizer, and accuracy as the evaluation metric.
Fit the Model: Train the model using the scaled training data.
Evaluate the Model: Evaluate the model using the test data to calculate the loss and accuracy.
Save the Model: Save the trained model as student_loans.keras and download the file from Colab for uploading to your GitHub repository.
Output: The trained neural network model, saved as student_loans.keras, and evaluation metrics (loss and accuracy).

### Part 3: Predict Loan Repayment Success Using the Neural Network Model
Objective: Use the saved neural network model to make predictions on the reserved test data and evaluate the performance.

Reload the Model: Load the saved student_loans.keras model.
Make Predictions: Use the model to predict outcomes on the test data, rounding the predictions to binary values (0 or 1).
Generate a Classification Report: Create a classification report that includes precision, recall, f1-score, and accuracy using the predictions and actual test labels.
Output: A classification report saved in classification_report.txt.

### Part 4: Discuss Creating a Recommendation System for Student Loans
Objective: Reflect on the creation of a recommendation system for student loans and answer the following questions:

## Conclusion
This project demonstrates how to preprocess data, design a neural network model, and evaluate its performance in predicting student loan repayment success. Additionally, the project explores the creation of a recommendation system for student loans, considering real-world challenges and appropriate filtering methods. The results of this project can help in designing predictive models that assist in evaluating loan repayment likelihood and recommending suitable loan options for students.






