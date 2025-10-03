import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

#------------------ Load Data ------------------
heart_data = pd.read_csv('heart_cleveland_upload.csv')

# لو فيه أعمدة زيادة (مثل Unnamed أو id) نحذفها
if 'Unnamed: 0' in heart_data.columns:
    heart_data = heart_data.drop(columns=['Unnamed: 0'], axis=1)

print("Columns:", heart_data.columns)
print("Shape:", heart_data.shape)

#------------------ Features / Target ------------------
X = heart_data.drop(columns='condition', axis=1)
Y = heart_data['condition']

print("Number of features:", X.shape[1])

#------------------ Split Data ------------------
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.2, stratify=Y, random_state=2
)

#------------------ Train Model ------------------
model = LogisticRegression(max_iter=1000)  # زيادة عدد iterations
model.fit(X_train, Y_train)

#------------------ Accuracy ------------------
X_train_prediction = model.predict(X_train)
training_data_accuracy = accuracy_score(Y_train, X_train_prediction)

print("Training Accuracy:", training_data_accuracy)

#------------------ Prediction for New Data ------------------
# لازم تدخل نفس عدد الأعمدة بالضبط (X.shape[1])
input_data = (55, 1, 2, 150, 300, 0, 1, 150, 1, 1.5, 2, 2, 2) #13

# نحولها لمصفوفة ونعيد تشكيلها
input_data_as_numpy_array = np.asarray(input_data)
input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)

prediction = model.predict(input_data_reshaped)

if prediction[0] == 0:
    print('The Person does NOT have Heart Disease')
else:
    print('The Person HAS Heart Disease')
