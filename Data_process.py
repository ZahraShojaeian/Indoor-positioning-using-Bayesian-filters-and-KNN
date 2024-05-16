import pandas as pd
import numpy as np
from joblib import dump
from sklearn.model_selection import train_test_split



 
# Preparing data 

data = pd.read_excel("data.xlsx")

## Splitting the data into training and testing sets (80% train, 20% test)

data_train1 , data_test1  = train_test_split(data.iloc[:, 0:3], test_size=0.2)
data_train2 , data_test2  = train_test_split(data.iloc[:, 3:6], test_size=0.2)
data_train3 , data_test3  = train_test_split(data.iloc[:, 6:9], test_size=0.2)
data_train4 , data_test4  = train_test_split(data.iloc[:, 9:12], test_size=0.2)
data_train5 , data_test5  = train_test_split(data.iloc[:, 12:15], test_size=0.2)

data_train2.columns = ['WiFi A', 'WiFi B', 'WiFi C']
data_train3.columns = ['WiFi A', 'WiFi B', 'WiFi C']
data_train4.columns = ['WiFi A', 'WiFi B', 'WiFi C']
data_train5.columns = ['WiFi A', 'WiFi B', 'WiFi C']

data_test2.columns=  ['WiFi A', 'WiFi B', 'WiFi C']
data_test3.columns=  ['WiFi A', 'WiFi B', 'WiFi C']
data_test4.columns=  ['WiFi A', 'WiFi B', 'WiFi C']
data_test5.columns=  ['WiFi A', 'WiFi B', 'WiFi C']


data_train1.reset_index(drop=True, inplace=True)
data_train2.reset_index(drop=True, inplace=True)
data_train3.reset_index(drop=True, inplace=True)
data_train4.reset_index(drop=True, inplace=True)
data_train5.reset_index(drop=True, inplace=True)

data_test1.reset_index(drop=True, inplace=True)
data_test2.reset_index(drop=True, inplace=True)
data_test3.reset_index(drop=True, inplace=True)
data_test4.reset_index(drop=True, inplace=True)
data_test5.reset_index(drop=True, inplace=True)

# Prepparing data for train and test

# Train data

data1= data_train1.replace("NM", -100).infer_objects()
data2= data_train2.replace("NM", -100).infer_objects()
data4= data_train4.replace("NM", -100).infer_objects()
data5= data_train5.replace("NM", -100).infer_objects()

## Removing samples containing NM from location3's data 
data3= data_train3[data_train3.ne('NM').all(axis=1)]


# Test data

data_test1= data_test1.replace("NM", -100).infer_objects()
data_test2= data_test2.replace("NM", -100).infer_objects()
data_test3= data_test3.replace("NM", -100).infer_objects()
data_test4= data_test4.replace("NM", -100).infer_objects()
data_test5= data_test5.replace("NM", -100).infer_objects()
X_test= np.vstack((data_test1, data_test2, data_test3, data_test4, data_test5))





# Calcualting the likelihood function 

## APs Tables 

W_1= [data1['WiFi A'],data2['WiFi A'],data3['WiFi A'],data4['WiFi A'],data5['WiFi A']]
W_2= [data1['WiFi B'],data2['WiFi B'],data3['WiFi B'],data4['WiFi B'],data5['WiFi B']]
W_3= [data1['WiFi C'],data2['WiFi C'],data3['WiFi C'],data4['WiFi C'],data5['WiFi C']]
Wr= [W_1,W_2,W_3]
pdf = np.zeros((90, 3, 5))

## Fit data to a Gaussian distribution
for r in range(3):
    for l in range(5):
    # Calculate mean and standard deviation
        mu = np.mean(Wr[r][l])
        sigma = np.std(Wr[r][l])

    # Create Gaussian PDF
        x = np.arange(-100, -10)  # Range of values from -100 to -9
        pdf[:, r, l] = np.exp(-0.5 * ((x - mu) / sigma ) ** 2) / (sigma * np.sqrt(2 * np.pi))

pdf[:, 0, 4]=0
pdf[0, 0, 4]=1
np.save("pdf.npy", pdf)




# training KNN=3 model 

X_train= np.vstack((data1,data2,data4,data5,data3)) 
Y_train = np.array([[0,0,0,0,0,0,0,0,0,0,0,0,
                     1,1,1,1,1,1,1,1,1,1,1,1,
                     3,3,3,3,3,3,3,3,3,3,3,3,
                     4,4,4,4,4,4,4,4,4,4,4,4,
                     2,2,2,2,2,2,2,2,2,2,2,2]])
Y_train = Y_train.flatten()  # Flatten to 1D array


class KNN:
    def __init__(self, k):
        self.k = k

    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train

    def euclidean_distance(self, x1, x2):
        return np.sqrt(np.sum((x1 - x2) ** 2))

    def predict(self, X_test):
        y_pred = [self._predict(x) for x in X_test]
        return np.array(y_pred)

    def _predict(self, x):
        # Calculate distances between x and all examples in the training set
        distances = [self.euclidean_distance(x, x_train) for x_train in self.X_train]
        # Sort by distance and return indices of the first k neighbors
        k_indices = np.argsort(distances)[:self.k]
        # Extract the labels of the k nearest neighbor training samples
        k_nearest_labels = [self.y_train[i] for i in k_indices]
        # Return the most common class label
        most_common = np.argmax(np.bincount(k_nearest_labels))
        return most_common


## Initialize and fit the model
knn = KNN(k=3)
knn.fit(X_train, Y_train)

## Saving the KNN model
knn_model_file = "knn_model.joblib"
dump(knn, knn_model_file)




