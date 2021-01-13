from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression as LR
import numpy as np

import warnings

from bayes_opt import BayesianOptimization
from bayes_opt.util import Colours

class BayesOpt:
    
    def __init__(self,X_train, y_train, folds=5, log_scaling=True,score='accuracy', n_iter=10, init_points=5):
        self.X_train = X_train
        self.y_train = y_train
        self.folds = folds
        self.log_scaling = log_scaling
        self.score = score
        self.n_iter = n_iter
        self.init_points = init_points
        self.values = None
        
    def optimize_svm(self, values, kernel=None):
                  
        self.values = values
        
        if kernel is None:
            k=['linear', 'poly', 'rbf', 'sigmoid']
        elif type(kernel) == list:
            k=kernel
        else:
            k=[kernel]

        i=0
        maximum_opt = np.empty_like(k,dtype = dict)
        for ker_type in k:
            print ('Optimizing for {} kernel'.format(ker_type))
            maximum_opt[i] = self._optimize_svm_main(ker_type)
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

    def optimize_lr(self, values, penalty=None):
        
        self.values = values

        if penalty is None:
            norm=['l1', 'l2']
        elif type(penalty) == list:
            norm=penalty
        else:
            norm=[penalty]
        i=0
        maximum_opt = np.empty_like(norm,dtype = dict)
        for n in norm:
            print ('Optimizing for {} penalty'.format(n))
            maximum_opt[i] = self._optimize_lr_main(n)
            i=i+1

        maximum_opt = maximum_opt.reshape(len(norm),1)
        maxi = 0
        maxi_dict = {}
        for i in range(len(norm)):
            if maximum_opt[i][0]['target'] > maxi:
                maxi = maximum_opt[i][0]['target']
                maxi_dict = maximum_opt[i][0]
                maxi_dict['params']['penalty']=norm[i] 

        print('\n')
        print("Final result: The optimal model's accuracy is {} and the optimal parameters are C={} and penalty={}". format(maxi_dict['target'],maxi_dict['params']['C'],maxi_dict['params']['penalty']))
        return maxi_dict
    
    def optimize_rf(self, values):

        self.values = values
        
        maxi_dict = self._optimize_rf_main()
        
        maxi_dict['params']['n_estimators'] = int(maxi_dict['params']['n_estimators'])
        
        if maxi_dict['params']['min_samples_split']>1:
            maxi_dict['params']['min_samples_split']=int(maxi_dict['params']['min_samples_split'])
        print('\n')
        print("Final result: The optimal model's score is {} and the optimal parameters are n_estimators={}, min_samples_split={} and max_features={}". format(maxi_dict['target'],maxi_dict['params']['n_estimators'],maxi_dict['params']['min_samples_split'],maxi_dict['params']['max_features']))
        return maxi_dict


    def _svm_cv(self, C, gamma, kernel):
        """SVC cross validation.
        This function will instantiate a SVC classifier with parameters C and
        gamma. Combined with data and targets this will in turn be used to perform
        cross validation. The result of cross validation is returned.
        Our goal is to find combinations of C and gamma that maximizes the roc_auc
        metric.
        """
        estimator = SVC(C=C, gamma=gamma, kernel=kernel, random_state=42)
        cval = cross_val_score(estimator, self.X_train, self.y_train, scoring=self.score,cv=self.folds)
        return cval.mean()

    def _optimize_svm_main(self, kernel):
        """Apply Bayesian Optimization to SVC parameters."""
        def svm_crossval(C, gamma):
            """Wrapper of SVC cross validation.
            Notice how we transform between regular and log scale. While this
            is not technically necessary, it greatly improves the performance
            of the optimizer.
            """
            
            if self.log_scaling:
                C = 10 ** C
                gamma = 10 ** gamma
            
            return self._svm_cv(C=C, gamma=gamma, kernel=kernel)
        
        optimizer = BayesianOptimization(
            f=svm_crossval,
            pbounds=self.values,
            random_state=42,
            verbose=2
        )
        
        try:
            optimizer.maximize(n_iter=self.n_iter, init_points = self.init_points)
        except:
            print('Error related to scaling.')

        return optimizer.max
    
    def _lr_cv(self, C, penalty):
        """lr cross validation.
        This function will instantiate a lr classifier with parameters C and
        gamma. Combined with data and targets this will in turn be used to perform
        cross validation. The result of cross validation is returned.
        Our goal is to find combinations of C and gamma that maximizes the roc_auc
        metric.
        """
        estimator = LR(C=C, penalty=penalty, random_state=42)
        cval = cross_val_score(estimator, self.X_train, self.y_train, cv=self.folds)
        return cval.mean()

    def _optimize_lr_main(self, n):
        """Apply Bayesian Optimization to lr parameters."""
        def lr_crossval(C):
            """Wrapper of lr cross validation.
            Notice how we transform between regular and log scale. While this
            is not technically necessary, it greatly improves the performance
            of the optimizer.
            """
                
            if self.log_scaling:
                C = 10 ** C
            
            return self._lr_cv(C=C, penalty=n)

        optimizer = BayesianOptimization(
            f=lr_crossval,
            pbounds=self.values,
            random_state=42,
            verbose=2
        )
        
        try:
            optimizer.maximize(n_iter=self.n_iter, init_points = self.init_points)
        except:
            print('Error related to scaling.')
        return optimizer.max

    
    def _rf_cv(self, n_estimators, min_samples_split, max_features):
        """
        Random Forest Cross Validation.
        This function will instantiate a random forest classifier with parameters
        n_estimators, min_samples_split, and max_features. Combined with data and
        targets this will in turn be used to perform cross validation. The result
        of cross validation is returned.
        Our goal is to find combinations of n_estimators, min_samples_split, and
        max_features that minimzes the log loss.
        """

        
        estimator = RFC(n_estimators=n_estimators, min_samples_split=min_samples_split, max_features=max_features, random_state=42)
        cval = cross_val_score(estimator, self.X_train, self.y_train, cv=self.folds)
        
        return cval.mean()

    def _optimize_rf_main(self):
        """Apply Bayesian Optimization to rf parameters."""
        def rf_crossval(n_estimators, min_samples_split, max_features):
            """Wrapper of rf cross validation.
            Notice how we transform between regular and log scale. While this
            is not technically necessary, it greatly improves the performance
            of the optimizer.
            """
            if min_samples_split<1:
                return self._rf_cv(
                    int(n_estimators),
                    min_samples_split,
                    max_features,
                )
            else:
                return self._rf_cv(
                    int(n_estimators),
                    int(min_samples_split),
                    max_features,
                )
                
        optimizer = BayesianOptimization(
            f=rf_crossval,
            pbounds=self.values,
            random_state=42,
            verbose=2
        )
        
        try:
            optimizer.maximize(n_iter=self.n_iter, init_points = self.init_points)
        except:
            print('Error related to scaling.')
        return optimizer.max