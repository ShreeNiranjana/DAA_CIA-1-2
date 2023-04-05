#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from pyswarms.single import GlobalBestPSO
import time


# In[ ]:


df = pd.read_csv("Bank_Personal_Loan_Modelling.csv")
df = df = df[['Age', 'Experience', 'Income', 'ZIP Code', 'Family', 'CCAvg',
       'Education', 'Mortgage', 'Securities Account',
       'CD Account', 'Online', 'CreditCard', 'Personal Loan']]


# In[ ]:


X = df.iloc[:,:-1].values
Y = df.iloc[:,-1].values
n_features = X.shape[1]
n_classes = len(df['Personal Loan'].unique())


# In[ ]:


sc = StandardScaler()
X = sc.fit_transform(X)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=32)


# In[ ]:


def create_custom_model(input_dim, output_dim, nodes, n=1, name='model'):
    model = Sequential(name=name)
    for i in range(n):
        model.add(Dense(nodes, input_dim=input_dim, activation='relu'))
    
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
    return model


# In[ ]:


n_layers = 2
model = create_custom_model(n_features, n_classes,4, n_layers)
model.summary()

callback = model.fit(X_train, Y_train,batch_size=5,epochs=50,validation_data=(X_test, Y_test))
score = model.evaluate(X_test, Y_test)

print('Test loss:', score[0])
print('Test accuracy:', score[1])


# In[ ]:


def get_shape(model):
    weights_layer = model.get_weights()
    shapes = []
    for weights in weights_layer:
        shapes.append(weights.shape)
    return shapes
def set_shape(weights,shapes):
    new_weights = []
    index=0
    for shape in shapes:
        if(len(shape)>1):
            n_nodes = np.prod(shape)+index
        else:
            n_nodes=shape[0]+index
        tmp = np.array(weights[index:n_nodes]).reshape(shape)
        new_weights.append(tmp)
        index=n_nodes
    return new_weights


# In[ ]:


def evaluate_nn(W, shape,X_train=X_train, Y_train=Y_train):
    results = []
    for weights in W:
        model.set_weights(set_shape(weights,shape))
        score = model.evaluate(X_train, Y_train, verbose=0)
        results.append(1-score[1])
    return results


# In[ ]:


shape = get_shape(model)
x_max = 1.0 * np.ones(83)
x_min = -1.0 * x_max
bounds = (x_min, x_max)
options = {'c1': 0.4, 'c2': 0.8, 'w': 0.4}
optimizer = GlobalBestPSO(n_particles=25, dimensions=83,options=options, bounds=bounds)


# In[ ]:


cost, pos = optimizer.optimize(evaluate_nn, 15, X_train=X_train, Y_train=Y_train,shape=shape)
model.set_weights(set_shape(pos,shape))
score = model.evaluate(X_test, Y_test)
print('Test loss:', score[0])
print('Test accuracy:', score[1])


# In[ ]:




