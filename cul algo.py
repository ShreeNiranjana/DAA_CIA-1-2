#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
data = pd.read_csv(r"C:\Users\niran\Downloads\Bank_Personal_Loan_Modelling.csv")
data.head()


# In[ ]:


data = data.drop(columns = ["ID","ZIP Code"])
X = data.drop("Personal Loan",axis=1)
y = data["Personal Loan"]

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=0)
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
import keras
from keras.models import Sequential
from keras.layers import Dense

from keras.optimizers import Adam

optimizer = Adam(lr=0.01)
loss = 'binary_crossentropy'


# In[ ]:


def fitness(weights):
    w0 = weights[:66].reshape((11, 6))
    w1 = weights[66:72].reshape((6,1))
    b0= np.array([0.,0.,0.,0.,0.,0.])
    b1 = np.array([0.])


    model.layers[0].set_weights([w0, b0])
    model.layers[1].set_weights([w1, b1])
    
    
    model.compile(optimizer=optimizer, loss=loss)
    
    Loss = model.evaluate(X_train, y_train, verbose=0)
    return -Loss


# In[ ]:


def cultural_algorithm(population_size, generations, mutation_rate, belief_space_size):
    
    # Initialize the population with random weights
    population = [np.random.uniform(low=-1, high=1, size=77) for _ in range(population_size)]
    belief_space = [np.random.uniform(low=-1, high=1, size=77) for _ in range(belief_space_size)]

    for generation in range(generations):
        fitness_scores = [fitness(x) for x in population]

        indices = np.argsort(fitness_scores)[-2:]
        parents = [population[i] for i in indices]

        offspring = []
        for _ in range(population_size - len(parents)):
            parent1 = belief_space[np.random.randint(belief_space_size-1)].flatten() #Choose randomly from belief space
            parent2 = parents[1].flatten()

            crossover_point = np.random.randint(0, len(parent1))
            
            #Crossover 
            child = np.concatenate([parent1[:crossover_point], parent2[crossover_point:]])
            child = child.reshape(parents[0].shape)

            # Perform mutation
            for i in range(len(child)):
                if np.random.uniform() < mutation_rate:
                    child[i] += np.random.normal(loc=0, scale=0.1)

            offspring.append(child)

        population = parents + offspring
        
        #Update the belief space by choosing the best parents and sorting them
        belief_space = sorted(belief_space + parents, key=lambda x: fitness(x), reverse=True)[:belief_space_size]
        
        if (generation%5==0):
            print("Finished Generation:",generation)

    fitness_scores = [fitness(x) for x in population]
    best_index = np.argmax(fitness_scores)
    best_weights = population[best_index]

    return best_weights


# In[ ]:


model = Sequential()
model.add(Dense(units=6, activation='relu', input_dim=X_train.shape[1]))
model.add(Dense(units=1, activation='sigmoid'))
best_weights = cultural_algorithm(population_size=20, generations=50, mutation_rate=0.1,belief_space_size = 10)
w0 = best_weights[:66].reshape((11, 6))
w1 = best_weights[66:72].reshape((6,1))
b0= np.array([0.,0.,0.,0.,0.,0.])
b1 = np.array([0.])

model.layers[0].set_weights([w0, b0])
model.layers[1].set_weights([w1, b1])

model.compile(optimizer=optimizer, loss=loss)


# In[ ]:


y_pred = model.predict(X_test)
y_pred = (y_pred < 0.00005)


# In[ ]:


from sklearn.metrics import accuracy_score
accuracy_score(y_test,y_pred)

