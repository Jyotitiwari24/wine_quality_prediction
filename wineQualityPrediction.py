import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier

wine_dataset = pd.read_csv('/content/winequality-red.csv')

wine_dataset.head()

wine_dataset.shape

wine_dataset.isnull().sum()

wine_dataset.describe()

type(wine_dataset)

# Number of values for each quality
sns.catplot(x = 'quality' , data  = wine_dataset ,  kind ='count')

# voltile acidity vs quality
plot = plt.figure(figsize= (5,5))
sns.barplot(x = 'quality', y = 'volatile acidity' , data = wine_dataset)

# Citric acid vs Quality
plot = plt.figure(figsize= (5,5))
sns.barplot(x = 'quality' , y = 'citric acid' , data = wine_dataset)

correlation = wine_dataset.corr()

# Constructing a heatmap to understand the correlation between the columns
plt.figure(figsize=(10,10))
sns.heatmap(correlation , cbar = True , square= True, fmt = '1f' , annot= True , annot_kws={'size': 8}, cmap='Blues')

# separate the data and label
X = wine_dataset.drop('quality' , axis = 1)

print(X)

Y = wine_dataset['quality'].apply(lambda  y_value: 1 if y_value >= 7 else 0)

print(Y)

X_train , X_test , Y_train , Y_test = train_test_split(X,Y , test_size=0.2 , random_state= 2)

print(X.shape ,X_test.shape, X_train.shape)

model = RandomForestClassifier()

# Training the Model with X_train
model.fit(X_train, Y_train)

X_test_prediction = model.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction, Y_test)

print('Accuracy is :', test_data_accuracy)

input_data = (7.5,0.5,0.36,6.1,0.071,17.0,102.0,0.9978,3.35,0.8,10.5)

# Changing the input data to numpy array
input_data_as_numpy_array = np.asarray(input_data)

# Reshape the data to predict label for one instance
input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)

prediction = model.predict(input_data_reshaped)
print(prediction)

if(prediction[0] == 1):
  print('Good Quality Wine')
else:
    print('Bad Quality Wine')


