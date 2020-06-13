#!/usr/bin/env python
# coding: utf-8

# In[2]:


def kennardstonealgorithm(df,endpoint, k):
    # Change the imported df in case it is not numpy
    import pandas as pd
    import numpy as np
    pan_df = df
    df = np.array(df)
    # Save the original df
    original_df = df
    k=len(df)-k
    maxi = 0
    for i in range(len(df)):
        for j in range(i,len(df)):
            if i!=j:
                DTA = ((df[i] - df[j])**2).sum()
                if DTA>maxi:
                    maxi=DTA
                    x0=i
                    y0=j            
    # Distance to average
    # List with the indexes of the testing dataset
    train_list = list()
    train_list.append(x0)
    train_list.append(y0)
    # Array of the train indexes
    test_list = pan_df.index
    df = np.delete(df, train_list, 0)
    test_list = np.delete(test_list, train_list, 0)
    for iteration in range(1,k-1):
        selected_samples = original_df[train_list, :]
        min_distance_to_selected_samples = list()
        for min_distance_calculation_number in range(len(df)):
            distance_to_selected_samples = ((selected_samples - np.tile(df[min_distance_calculation_number, :],(selected_samples.shape[0], 1))) ** 2).sum(axis=1)
            min_distance_to_selected_samples.append(np.min(distance_to_selected_samples))
        
        max_distance = np.where(min_distance_to_selected_samples == np.max(min_distance_to_selected_samples))
        max_distance = max_distance[0][0]
        train_list.append(test_list[max_distance])
        df = np.delete(df, max_distance, 0)
        test_list = np.delete(test_list, max_distance, 0)
        
    train_df = pan_df.drop(pan_df.index[test_list])
    test_df = pan_df.drop(pan_df.index[train_list])
    train_labels = train_df[endpoint]
    test_labels = test_df[endpoint]
    train_df.drop(endpoint, axis = 1, inplace=True)
    test_df.drop(endpoint, axis = 1, inplace=True)
    
    return train_df, test_df, train_labels, test_labels

