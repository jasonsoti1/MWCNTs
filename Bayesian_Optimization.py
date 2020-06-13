#!/usr/bin/env python
# coding: utf-8

# In[ ]:


def optimize_svm(train_df, train_labels, test_df, test_labels, dic, n_iter, init_points,kernel=None):
    import numpy as np
    if type(kernel) == list:
        k=kernel
    else:
        if kernel==None:
            k=['linear', 'poly', 'rbf', 'sigmoid']
        else:
            k=[kernel]
    
    i=0
    maximum_opt = np.empty_like(k,dtype = dict)
    for ker_type in k:
        print ('Optimizing for {} kernel'.format(ker_type))
        maximum_opt[i] = optimize_main_svm(train_df, train_labels, test_df, test_labels, dic, n_iter, init_points, ker_type)
        i=i+1
    
    maximum_opt = maximum_opt.reshape(len(k),1)
    maxi = 0
    maxi_dict = {}
    for i in range(len(k)):
        if maximum_opt[i][0]['target'] > maxi:
            maxi = maximum_opt[i][0]['target']
            maxi_dict = maximum_opt[i][0]
            maxi_dict['params']['kernel']=k[i] 
        
    print('\n')
    print("Final result: The optimal model's accuracy is {} and the optimal parameters are C={}, gamma={} and kernel={}". format(maxi_dict['target'],maxi_dict['params']['C'],maxi_dict['params']['gamma'],maxi_dict['params']['kernel']))
    return maxi_dict

def optimize_rfc(train_df, train_labels, test_df, test_labels, dic, n_iter, init_points):
    maxi_dict={}
    maxi_dict = optimize_main_rfc(train_df, train_labels, test_df, test_labels, dic, n_iter, init_points)
    maxi_dict['params']['n_estimators'] = int(maxi_dict['params']['n_estimators'])
    if maxi_dict['params']['min_samples_split']>1:
        maxi_dict['params']['min_samples_split']=int(maxi_dict['params']['min_samples_split'])
    print('\n')
    print("Final result: The optimal model's accuracy is {} and the optimal parameters are n_estimators={}, min_samples_split={} and max_features={}". format(maxi_dict['target'],maxi_dict['params']['n_estimators'],maxi_dict['params']['min_samples_split'],maxi_dict['params']['max_features']))
    return maxi_dict
    
def optimize_lr(train_df, train_labels, test_df, test_labels, dic, n_iter, init_points, norm=None):
    import numpy as np
    
    if type(norm) == list:
        norm=norm
    else:
        if norm==None:
            norm=['l1','l2']
        else:
            norm=[norm]
    i=0
    maximum_opt = np.empty_like(norm,dtype = dict)
    for n in norm:
        print ('Optimizing for {} norm'.format(n))
        maximum_opt[i] = optimize_main_lr(train_df, train_labels, test_df, test_labels, dic, n_iter, init_points, n)
        i=i+1
    
    maximum_opt = maximum_opt.reshape(len(norm),1)
    maxi = 0
    maxi_dict = {}
    for i in range(len(norm)):
        if maximum_opt[i][0]['target'] > maxi:
            maxi = maximum_opt[i][0]['target']
            maxi_dict = maximum_opt[i][0]
            maxi_dict['params']['norm']=norm[i] 
        
    print('\n')
    print("Final result: The optimal model's accuracy is {} and the optimal parameters are C={} and penalty={}". format(maxi_dict['target'],maxi_dict['params']['C'],maxi_dict['params']['norm']))
    return maxi_dict

def optimize_pls(train_df, train_labels, test_df, test_labels, dic, n_iter, init_points):
    maxi_dict={}
    maxi_dict = optimize_main_pls(train_df, train_labels, test_df, test_labels, dic, n_iter, init_points)
    maxi_dict['params']['n_components'] = int(round(maxi_dict['params']['n_components'],0))
    print('\n')
    print("Final result: The optimal model's accuracy is {} and the optimal parameters are n_components={}". format(maxi_dict['target'],maxi_dict['params']['n_components']))
    return maxi_dict


# In[2]:


def optimize_main_svm(train_df, train_labels, test_df, test_labels, dic, n_iter, init_points, k):
    """Apply Bayesian Optimization to SVM parameters."""
    def svm_val(C, gamma):
        """Wrapper of SVM cross validation.
        Notice how we transform between regular and log scale. While this
        is not technically necessary, it greatly improves the performance
        of the optimizer.
        """
        return svm_cv(C, gamma, train_df, train_labels, test_df, test_labels,k)

    from bayes_opt import BayesianOptimization
    optimizer = BayesianOptimization(
        f=svm_val,
        pbounds=dic,
        random_state=42,
        verbose=2
    )
    
    optimizer.maximize(n_iter=n_iter, init_points=init_points)

    return optimizer.max


# In[1]:


def svm_cv(C, gamma, train_df, train_labels, test_df, test_labels,k):
    """SVM cross validation.
    This function will instantiate a SVM classifier with parameters C and
    gamma. Combined with data and targets this will in turn be used to perform
    cross validation. The result of cross validation is returned.
    Our goal is to find combinations of C and gamma that maximizes the roc_auc
    metric.
    """
    from sklearn.svm import SVC
    from sklearn.metrics import accuracy_score
    import numpy as np

    estimator = SVC(C=C, gamma=gamma, kernel=k,random_state=42, decision_function_shape='ovo')
    estimator.fit(train_df,train_labels)
    predictions = estimator.predict(test_df)
    val=100*accuracy_score(test_labels, predictions)      

    return val


# In[4]:


def optimize_main_rfc(train_df, train_labels, test_df, test_labels, dic, n_iter, init_points):
    """Apply Bayesian Optimization to Random Forest parameters."""
    def rfc_val(n_estimators, min_samples_split, max_features):
        """Wrapper of RandomForest cross validation.
        Notice how we ensure n_estimators and min_samples_split are casted
        to integer before we pass them along. Moreover, to avoid max_features
        taking values outside the (0, 1) range, we also ensure it is capped
        accordingly.
        """
        if min_samples_split<1:
            return rfc_cv(
                int(n_estimators),
                min_samples_split,
                max_features,
                train_df, train_labels, test_df, test_labels,
            )
        else:
            return rfc_cv(
                int(n_estimators),
                int(min_samples_split),
                max_features,
                train_df, train_labels, test_df, test_labels,
            )
            
    
    from bayes_opt import BayesianOptimization
    optimizer = BayesianOptimization(
        f=rfc_val,
        pbounds=dic,
        random_state=42,
        verbose=2
    )
    
    optimizer.maximize(n_iter=n_iter,init_points=init_points)
    
    return optimizer.max


# In[5]:


def rfc_cv(n_estimators, min_samples_split, max_features, train_df, train_labels, test_df, test_labels):
    """Random Forest cross validation.
    This function will instantiate a random forest classifier with parameters
    n_estimators, min_samples_split, and max_features. Combined with data and
    targets this will in turn be used to perform cross validation. The result
    of cross validation is returned.
    Our goal is to find combinations of n_estimators, min_samples_split, and
    max_features that minimzes the log loss.
    """
    
    from sklearn.ensemble import RandomForestClassifier as RFC
    from sklearn.metrics import accuracy_score
    import numpy as np

    estimator = RFC(n_estimators=n_estimators, min_samples_split=min_samples_split, max_features=max_features, random_state=42)
    estimator.fit(train_df,train_labels)
    predictions = estimator.predict(test_df)
    
    val=100*accuracy_score(test_labels, predictions)
        
    return val


# In[6]:


def optimize_main_lr(train_df, train_labels, test_df, test_labels, dic, n_iter, init_points,n):
    """Apply Bayesian Optimization to Random Forest parameters."""
    def lr_val(C):
        """Wrapper of RandomForest cross validation.
        Notice how we ensure n_estimators and min_samples_split are casted
        to integer before we pass them along. Moreover, to avoid max_features
        taking values outside the (0, 1) range, we also ensure it is capped
        accordingly.
        """
        return lr_cv(C,train_df, train_labels, test_df, test_labels,n)
    
    from bayes_opt import BayesianOptimization
    optimizer = BayesianOptimization(
        f=lr_val,
        pbounds=dic,
        random_state=42,
        verbose=2
    )
    
    optimizer.maximize(n_iter=n_iter,init_points=init_points)
    
    return optimizer.max


# In[7]:


def lr_cv(C, train_df, train_labels, test_df, test_labels,n):
    """Random Forest cross validation.
    This function will instantiate a random forest classifier with parameters
    n_estimators, min_samples_split, and max_features. Combined with data and
    targets this will in turn be used to perform cross validation. The result
    of cross validation is returned.
    Our goal is to find combinations of n_estimators, min_samples_split, and
    max_features that minimzes the log loss.
    """
    
    from sklearn.linear_model import LogisticRegression as LR
    from sklearn.metrics import accuracy_score
    import numpy as np

    estimator = LR(C=C, penalty=n, random_state=42)
    estimator.fit(train_df,train_labels)
    predictions = estimator.predict(test_df)
    
    val=100*accuracy_score(test_labels, predictions)
        
    return val


# In[8]:


def optimize_main_pls(train_df, train_labels, test_df, test_labels, dic, n_iter, init_points):
    """Apply Bayesian Optimization to Random Forest parameters."""
    def pls_val(n_components):
        """Wrapper of RandomForest cross validation.
        Notice how we ensure n_estimators and min_samples_split are casted
        to integer before we pass them along. Moreover, to avoid max_features
        taking values outside the (0, 1) range, we also ensure it is capped
        accordingly.
        """
        return pls_cv(int(n_components),train_df, train_labels, test_df, test_labels)
    
    from bayes_opt import BayesianOptimization
    optimizer = BayesianOptimization(
        f=pls_val,
        pbounds=dic,
        random_state=42,
        verbose=2
    )
    
    optimizer.maximize(n_iter=n_iter,init_points=init_points)
    
    return optimizer.max


# In[1]:


def pls_cv(n_components, train_df, train_labels, test_df, test_labels):
    """Random Forest cross validation.
    This function will instantiate a random forest classifier with parameters
    n_estimators, min_samples_split, and max_features. Combined with data and
    targets this will in turn be used to perform cross validation. The result
    of cross validation is returned.
    Our goal is to find combinations of n_estimators, min_samples_split, and
    max_features that minimzes the log loss.
    """
    
    from sklearn.cross_decomposition import PLSRegression as PLS
    from sklearn.metrics import accuracy_score
    import numpy as np

    estimator = PLS(n_components=n_components, scale=False)
    estimator.fit(train_df,train_labels)
    predictions = estimator.predict(test_df)
    predictions = predictions.reshape(len(predictions))
    for i in range(len(predictions)):
        predictions[i]  = round(predictions[i], 0)
        
    val=100*accuracy_score(test_labels, predictions)
  
    return val


# In[ ]:




