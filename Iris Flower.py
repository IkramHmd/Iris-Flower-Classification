# Import Libraries 
import pandas as pd ;
import numpy as np ; 
import matplotlib.pyplot as plt ; 
import seaborn as sns  ; 
'''- Split your data into training and testing sets (`train_test_split`).
- Evaluate the performance of your model with accuracy and detailed classification metrics (`accuracy_score` and `classification_report`).
- Use the K-Nearest Neighbors algorithm for classification tasks (`KNeighborsClassifier`). 
'''
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.neighbors import KNeighborsClassifier

from sklearn.datasets import load_iris
data = load_iris()
df = pd.DataFrame(data.data, columns=data.feature_names)
df['species'] = data.target
iris = pd.read_csv("https://raw.githubusercontent.com/amankharwal/Website-data/master/IRIS.csv") ; 
print(iris.head()) ; 
print(iris.describe()) ; 
#target labels: 
print("Target Labels", iris["species"].unique()) ; 
#plot the iris species 
import plotly.express as px
fig = px.scatter(iris, x="sepal_width", y="sepal_length", color="species")
fig.show() ; 

# Separate features and target labels
X = iris.drop(columns='species')  # Features (sepal width, sepal length, etc.)
y = iris['species']                # Target labels (species)
# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Create the KNN classifier
knn = KNeighborsClassifier(n_neighbors=3)  
# Train the KNN classifier
knn.fit(X_train, y_train)
# Make predictions on the test set
y_pred = knn.predict(X_test)
# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy) 
# Detailed classification report
print(classification_report(y_test, y_pred))
# Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=data.target_names, yticklabels=data.target_names)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show() ; 

