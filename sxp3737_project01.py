#!/usr/bin/env python
# coding: utf-8

# In[326]:


#importing 
import numpy as np
import pandas as pd
import statistics


# In[327]:


#opening the dataset
df= pd.read_csv('iris.data', names=['attr01', 'attr02', 'attr03', 'attr04', 'class'],header=None)


# In[328]:


#Shuffling randomly the rows of the dataset
#source: https://www.kite.com/python/answers/how-to-shuffle-the-rows-in-a-pandas-dataframe-in-python
df = df.sample(frac=1).reset_index(drop=True)


# In[329]:


#splitting the dataset into source and target variables
#source: https://www.shanelynn.ie/using-pandas-dataframe-creating-editing-viewing-data-in-python/#:~:text=To%20delete%20rows%20and%20columns,the%20%E2%80%9Caxis%E2%80%9D%20as%201
source_data = df.drop('class', axis= 1)
target_column = df['class']


# In[330]:


# Relacing the String to Integer  in the target class data
#source: https://www.geeksforgeeks.org/replacing-strings-with-numbers-in-python-for-data-analysis/
unique_values = dict([(y, x + 1) for x, y in enumerate(sorted(set(target_column)))])
target_dframe = pd.DataFrame([unique_values[item] for item in target_column])


# In[331]:


#function to X_train, Y_train, X_test, Y_test according to the k for cross validation\
def split_train_test(split_data):
    var_test = []
    var_train = []
    for a in range(len(split_data)):
        var_intermediate_train = []
        for b in range(len(split_data)):
            if a==b:
                #adding to test set
                var_test.append(split_data[b])
            else:
                #adding to training set
                var_intermediate_train.append(split_data[b])
        var_train.append(var_intermediate_train)
    training_set = []
    for i in var_train:
        training_set.append(np.matrix(pd.concat(i)))
        
    return training_set, var_test
                   
                    
            
    


# In[332]:


#function for linear regression and k fold cross validation
def linear_reg(k_folds):
    #splitting our source an target data in to k partitions
    split_source = np.array_split(source_data, k_folds)
    split_target = np.array_split(target_dframe, k_folds)
    
    #initializing X_train, Y_train, X_test, Y_test according to the k fol cross validation
    X_train, X_test = split_train_test(split_source)
    Y_train, Y_test = split_train_test(split_target)
    
    #as there are 4 independent variables it will be multivariate regression(more than one independent variable)
    #source: https://cmdlinetips.com/2020/03/linear-regression-using-matrix-multiplication-in-python-using-numpy/
    #calculating our beta cap for each fold partion matrix
    beta_k= []
    for k in range(len(X_train)):  
        beta_k.append(np.linalg.inv(X_train[k].T.dot(X_train[k])).dot(X_train[k].T).dot(Y_train[k]))
        
    #getting the Y predict from the X test and the beta coeffiecent
    y_predict = []
    for k_th in range(len(X_test)):
        y_predict.append(np.matrix(X_test[k_th]).dot(np.matrix(beta_k[k_th])))    
        acc, rmse =calc_accuracy(y_predict , Y_test)
    print("RMSE for this iteration")
    c=0.0
    for i in range(len(rmse)):
        c=c+ rmse[i]
    print(str(c/k_folds))
    return statistics.mean(acc)


# In[333]:


##function for calculating the accuracy
def calc_accuracy(y_predict1 , Y_test1):
   
    Y_testing = []
    total_match = []
    accuracy = []
    rmse= []
     #converting each fold partition Y test data to list
    for i in range(len(Y_test1)):
        Y_testing.append(list(Y_test1[i][0]))
    
    predict_y=[]
    #rounding off the float value  to integer
    for p in y_predict1:
        predict_y.append(np.around(p))
    
    #Comparing the actual and predicted values
    for i in range(len(predict_y)):
        per_fold_match = 0
        
        for j in range(len(predict_y[i])):
            rmse_sum = sum((Y_testing[i][j] - y_predict1[i][j]) ** 2)
            if(predict_y[i][j] == Y_testing[i][j]):
                per_fold_match =  per_fold_match + 1
        accuracy.append(per_fold_match/len(predict_y[i]))
        rmse.append(np.sqrt(rmse_sum / len(predict_y[i])))
    
     
        
    return accuracy, rmse


# In[334]:


print("Accuracy when k= 3 the K fold cross validation %.2f" % (linear_reg(3)*100)+"%")


# In[335]:


print("Accuracy when k= 5 the K fold cross validation %.2f" % (linear_reg(5)*100)+"%")


# In[336]:


print("Accuracy when k= 7 the K fold cross validation %.2f" % (linear_reg(7)*100)+"%")


# In[337]:


print("Accuracy when k= 10 the K fold cross validation %.2f" % (linear_reg(12)*100)+"%")


# In[338]:


print("Accuracy when k= 15 the K fold cross validation %.2f" % (linear_reg(15)*100)+"%")

