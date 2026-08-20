import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score

diabetes = fetch_openml(name='diabetes',version=1,as_frame=False)

X = diabetes.data
y = diabetes.target

y = np.where(y =='tested_positive',1,0)

X_train,X_test,y_train,y_test = train_test_spilt(
	X , y, test_size = 0.2, random_state = 42)

def evaluate(y_true,y_pred):
	print("Accuracy :",accuracy_score)
