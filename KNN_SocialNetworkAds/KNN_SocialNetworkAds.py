"""

@author: UNISEPP
"""
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import plot_confusion_matrix

# importing the dataset
df = pd.read_csv('Social_Network_Ads.csv')

#selecting the dataset
X = df.loc[: , ['Age' , 'EstimatedSalary']]
y = df.iloc[:,4] #labels

# splitting the dataset
x_trn , x_tst , y_trn , y_tst = train_test_split(X , y , test_size= 0.3 , shuffle=True)

# standard scale
sc = StandardScaler()
x_trn = sc.fit_transform(x_trn)
x_tst = sc.fit_transform(x_tst)

# fitting the algorithm to the training dataset
knn = KNeighborsClassifier(n_neighbors = 5 )
knn.fit(x_trn,y_trn)

# predicting the train test result
predicted= knn.predict(x_tst)
print(predicted)
      
# calculating the accuracy
acc = knn.score(x_tst , y_tst)
print(acc)

# plotting the confusion matrix
cm = plot_confusion_matrix(knn , x_tst ,y_tst , normalize='true' )