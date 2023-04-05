#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


df = pd.read_csv(r'C:\Users\niran\Downloads\Bank_Personal_Loan_Modelling.csv')
df.head()


# In[3]:


df.nunique()


# In[4]:


df.drop(['ID'],inplace=True,axis=1)


# In[5]:


df.info()


# In[6]:


df['Age'] = pd.cut(df['Age'],bins=[23,30,45,67],labels=['Young','Adult','Old'])
df.head()


# In[7]:


df['Age'].value_counts()


# In[8]:


from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
df['Age'] = le.fit_transform(df['Age'])
df.columns


# In[20]:


col = ['Age', 'Experience', 'Income', 'ZIP Code', 'Family', 'CCAvg', 'Education', 
       'Mortgage', 'Securities Account','CD Account', 'Online', 'CreditCard', 'Personal Loan']
df = df[col]
x = df.iloc[:,:-1].values
y = df.iloc[:,-1].values


# In[ ]:


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.25)
from keras.models import Sequential
from keras.layers import Dense


# In[ ]:


keras = Sequential()
keras.add(Dense(12,input_dim=12,activation='relu'))
keras.add(Dense(8,activation='relu'))
keras.add(Dense(6,activation='relu'))


# In[ ]:


keras.add(Dense(1,activation='sigmoid'))
keras.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
keras.fit(x_train,y_train,epochs=20,batch_size=25)


# In[ ]:


_,acc = keras.evaluate(x_train,y_train)
print(acc*100)


# In[ ]:


y_pred = keras.predict(x_test)
y_pred = (y_pred>0.5)


# In[ ]:


from sklearn.metrics import accuracy_score
accuracy_score(y_test,y_pred)


# In[ ]:


from keras import backend as K
def r_loss(y_test,y_pred):
    res=K.sum(K.square(y_test-y_pred))    
    total=K.sum(K.square(y_test-K.mean(y_test)))
    return 1-(1-res/(total+K.epsilon()))


# In[ ]:


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        self.weights1 = np.random.randn(input_size, hidden_size)
        self.weights2 = np.random.randn(hidden_size, output_size)
    
    def forward(self, X):
        self.hidden = np.maximum(0, np.dot(X, self.weights1))
        self.output = np.dot(self.hidden, self.weights2)
        return self.output
    
    def backward(self, X, y, output):
        delta2 = output - y
        dweights2 = np.dot(self.hidden.T, delta2)
        delta1 = np.dot(delta2, self.weights2.T) * (self.hidden > 0)
        dweights1 = np.dot(X.T, delta1)
        return dweights1, dweights2
class AntColony:
    def __init__(self, n_ants, n_iterations, alpha, beta, rho, Q):
        self.n_ants = n_ants
        self.n_iterations = n_iterations
        self.alpha = alpha
        self.beta = beta
        self.rho = rho
        self.Q = Q
        
    def run(self, nn, X_train, y_train):
        pheromone = np.zeros(nn.weights1.shape)
        best_weights1 = nn.weights1
        best_weights2 = nn.weights2
        best_acc = 0
        
        for iteration in range(self.n_iterations):
            for ant in range(self.n_ants):
                weights1 = nn.weights1 + pheromone
                nn.weights1 = weights1
                output = nn.forward(X_train)
                acc = accuracy(output, y_train)
                dweights1, dweights2 = nn.backward(X_train, y_train, output)
                pheromone = self.rho * pheromone + self.Q * (dweights1 / acc)
                
                if acc > best_acc:
                    best_acc = acc
                     best_weights1 = nn.weights1
                    best_weights2 = nn.weights2
            
            nn.weights1 = best_weights1
            nn.weights2 = best_weights2
        
        return nn


# In[ ]:


def accuracy(y_pred, y_true):
    y_pred = np.argmax(y_pred, axis=1)
    return np.mean(y_pred == y_true)
nn = NeuralNetwork(X_train.shape[1], 100, 2)
aco = AntColony(n_ants=10, n_iterations=100, alpha=1, beta=2, rho=0.5, Q=100)

nn = aco.run(nn, X_train, y_train)
output = nn.forward(X_test)
acc = accuracy(output, y_test)
print(f'Test accuracy: {acc:.2f}')


# In[ ]:


X_train.shape


# In[ ]:


y_train.shape

