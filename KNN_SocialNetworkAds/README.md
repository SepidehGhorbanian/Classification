Predicting whether the client would buy a car or not

Using the KNN algorithm, Desicion Tree algorithm and Random Forest algorithm from Classification algorithms (Supervised)

The code is in Python and it is using the Scikit-learn (Sklearn) library

Dataset: The 'Social_Network_Ads' dataset
The dataset has these features: User ID, Gender, Age, EstimatedSalary, Purchased.
The 'Purchased' column is the lables and we put it in a 'y' variable.
We make the predictions based on these two features: Age, EstimatedSalary

KNN Algorithm: After preparing the dataset, we fit the KNN algorithm to the training dataset and then we can show the predicted labels for the testing dataset.
Lastly, we can calculate the accuracy and then plot the confusion matrix for better understanding the performance.

DecisionTree Algorithm: We can use the decision tree algorithm from sklearn to classify the training dataset. In this Decision Tree algorithm we want to choose the decision nods with the best gain and to reduce and decrease the entropy. Entropy measures data impurity and gain measures the redusction of the entropy.
We can use the 'predict' function to get predicted results for the testing dataset and use it for comparison against the actual testing labels.
Lastly we can use confusion matrix and calculate the performance.
