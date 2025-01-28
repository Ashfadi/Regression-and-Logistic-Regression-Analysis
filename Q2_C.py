#!/usr/bin/env python
# coding: utf-8

# In[2]:


def main():
    print('START Q2_C\n')
    
    # Importing Libraries and Dataset
    import numpy as np
    
    def clean_data(line):
        return line.replace('(', '').replace(')', '').replace(' ', '').strip().split(',')
    def fetch_data(filename):
        with open(filename, 'r') as f:
            input_data = f.readlines()
            clean_input = list(map(clean_data, input_data))
            f.close()
        return clean_input
    def readFile(dataset_path):
        input_data = fetch_data(dataset_path)
        input_np = np.array(input_data)
        return input_np

    training = './datasets/Q1_B_train.txt'
    test = './datasets/Q1_C_test.txt'
    Training_Data = readFile(training)
    Training_Data = Training_Data.astype(float)
    Test_Data = readFile(test)
    Test_Data = Test_Data.astype(float)
    
    #Assigning X,y values for Training and Test Data
    X = Training_Data[:,[0]]
    Y = Training_Data[:,[1]]
    Xtest = Test_Data[:,[0]]
    Ytest = Test_Data[:,[1]]
    
    # Weight Matrix
    def wm(point, X, gamma): 
        m = X.shape[0]    
      # Initialising w as an identity matrix
        w = np.mat(np.eye(m)) 
      # Calculating weights for all training examples [x(i)'s]
        for i in range(m): 
            xi = X[i] 
            d = (-2 * gamma * gamma) 
            w[i, i] = np.exp(np.dot((xi-point), (xi-point).T)/d)   
        return w
    
    #Function to predict values of Y by using Weight Matrix
    def predict(X, y, point, gamma): 
        m = X.shape[0] 
        X_ = np.append(X, np.ones(m).reshape(m,1), axis=1) 
        point_ = np.array([point, 1],dtype=object) 
        w = wm(point_, X_, gamma)   
      # Calculating parameter theta using the formula
        theta = np.linalg.pinv(X_.T*(w * X_))*(X_.T*(w * y)) 
      # Calculating predictions  
        pred = np.dot(point_, theta)  
        return theta, pred
    
    #Computing prediction and error by using locally weighted linear regression
    def predictions(X, y, gamma):
        X_test = Xtest 
        preds = [] 
        for point in X_test: 
            theta, pred = predict(X, Y, point, gamma) 
            preds.append(pred)
        error = np.sqrt(np.mean((preds - Ytest)**2))
        print('Data size = 128,','MSE =', error)
    predictions(X, Y, 0.204)
    
    ### Locally weighted regression learns a linear prediction that is only good locally, since far away errors do not weigh much in comparison to local ones.
    ### Even I got greater error than linear regression, the fit is good as it is shown in Q1_A.
    
    print('\nEND Q2_C\n')


if __name__ == "__main__":
    main()


# In[ ]:




