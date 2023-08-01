
#if __name__ == '__main__':

import statsmodels.api as sm
import train_test
#import scipy.stats
import pickle

#print(train_test.Y_train.values)
#print(train_test.X_train.values)

# GLM FIT-BINOMIAL
# =================

def GLM_Binomial_fit(Y_train, X_train):
    
    '''GLM Binomial fit'''
    X_train = X_train.values
    glm_binom = sm.GLM(Y_train, X_train, family=sm.families.Binomial())   
    res = glm_binom.fit()
    
    return res.summary(), res

m = (GLM_Binomial_fit(train_test.Y_train.values.reshape(-1,1), train_test.X_train))[1]
#print(m)

#print(m.predict(train_test.X_test.values))
#print(dir(m))

#p_val = scipy.stats.chi2.pdf(2356, 1957)
#print(p_val)

#pickle.dump(GLM_Binomial_fit, open('model.pkl','wb'))
