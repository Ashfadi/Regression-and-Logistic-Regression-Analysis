#!/usr/bin/env python
# coding: utf-8

# In[6]:


def main():
    print('START Q2_AB\n')
    
    # Importing Libraries and Dataset
    import numpy as np
    import matplotlib.pyplot as plt
    
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
    Training_Data = readFile(training)
    Training_Data = Training_Data.astype(float)
    
    #Assigning X,y values for Training and Test Data
    X = Training_Data[:,[0]]
    Y = Training_Data[:,[1]]
    
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
    
    #Plotting prediction by using locally weighted linear regression
    def plot_predictions(X, y, gamma, nval): #nval=size of test dataset
        X_test = X 
        preds = [] 
        for point in X_test: 
            theta, pred = predict(X, Y, point, gamma) 
            preds.append(pred)  
        X_test = np.array(X_test).reshape(nval,1)
        preds = np.array(preds).reshape(nval,1)
        plt.scatter(X, Y,label='Original')
        plt.plot(X_test, preds, 'r.',label='Predicted')
        plt.title("Locally Weighted Linear Regression")
        plt.xlabel("X-Value")
        plt.ylabel("Y-Value")
        plt.legend(loc = 'center left', bbox_to_anchor=(1, 0.5))
        plt.show()
    plot_predictions(X, Y, 0.204, 128)
    
    print('\nEND Q2_AB\n')


if __name__ == "__main__":
    main()


# In[ ]:




