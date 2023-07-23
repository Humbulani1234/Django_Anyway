
import unittest

import train_test
import pandas as pd
import class_diagnostics
import GLM_Bino

#print(dir(unittest.TestCase))

X_test = train_test.X_test 
X_train = train_test.X_train
#X_train = 2
Y_test = train_test.Y_test
Y_train = train_test.Y_train.to_frame()
#Y_train = 2
threshold = 0.47
function = GLM_Bino.GLM_Binomial_fit

class testing(unittest.TestCase):

    # def __init__(self, X_test):

    #     self.x_test = X_test

    def test_quantile_residuals(self): # start tyhe method name with the prefix test

        #result =
        #self.assertEqual()
        X_test = train_test.X_test 
        instance = class_diagnostics.Base(function, X_test, Y_test, X_train, Y_train, threshold)
        #y = instance.quantile_residuals()
        self.assertEqual(instance.quantile_residuals()[1], 1)
        self.assertIsInstance(X_test, pd.DataFrame)
        # self.assertIsInstance(instance.quantile_residuals()[0], pd.Series)
        # self.assertIsInstance(instance.quantile_residuals()[0], None)


if __name__ == "__main__":

    # just like main initialise variables and create relevant instances for testing
    
    unittest.main()