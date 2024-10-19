#Import the Required Libraries At the top of your code, you should import the necessary libraries.
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics


#Get the Data into the Program Load the Iris dataset into your program using pandas. You can download the dataset from the UCI repository and save it as iris.csv.
# Load the dataset
data = pd.read_csv('iris.csv')

# Verify that the data has been successfully imported
print(data.head())
print(data.info())

#Data Preprocessing We'll need to convert the species names into numerical values for the model to understand:
# Setosa - 0, Versicolor - 1, Virginica - 2
data["species"] = data["species"].replace({"setosa": 0, "versicolor": 1, "virginica": 2})

#Visualize the Data (Optional) To understand the relationships in our data, we can create some scatter plots. This step is optional and can be skipped if you're short on time.
# Graphs to visualize relationships
plt.subplot(221)
plt.scatter(data["petal_length"], data["species"], s=10, c='green', marker='o')
plt.subplot(222)
plt.scatter(data["petal_width"], data["species"], s=10, c='red', marker='o')
plt.subplot(223)
plt.scatter(data["sepal_length"], data["species"], s=10, c='blue', marker='o')
plt.subplot(224)
plt.scatter(data["sepal_width"], data["species"], s=10, c='orange', marker='o')
plt.show()

#Splitting the Data Next, we will split the data into features (X) and labels (Y). We'll use 80% of the data for training and 20% for testing.
Y = data["species"]
X = data.drop("species", axis=1)

# Split the data into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=1)

print("Training data shape:", X_train.shape)
print("Testing data shape:", X_test.shape)

#Model Selection and Training We'll use the Decision Tree Classifier for this project. Here’s how to create and train the model:
model = DecisionTreeClassifier(max_depth=3, random_state=1)
model.fit(X_train, Y_train)

#Making Predictions After training, we can use our model to make predictions on the test set
predictions = model.predict(X_test)


#Evaluating the Model Finally, we'll check the accuracy of our model:
print("Accuracy:", metrics.accuracy_score(predictions, Y_test))

