#!/usr/bin/env python
# coding: utf-8

# In[5]:


def main():
    print('START Q1_D\n')
    
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
    test = './datasets/Q1_C_test.txt'
    Training_Data = readFile(training)
    Training_Data = Training_Data.astype(float)
    Test_Data = readFile(test)
    Test_Data = Test_Data.astype(float)
    
    #Assigning X,y values for Training and Test Data
    #Taking first 20 elements of Training Data
    X = Training_Data[0:20,[0]]
    y = Training_Data[0:20,[1]]
    Xtest = Test_Data[:,[0]]
    ytest = Test_Data[:,[1]]
    
    #Function to calculate Coeffiecients
    def get_coefficients(X, k, d):
        x_new = np.ones((X.shape[0], 1))
        for i in range(1, d+1):
            x_new = np.append(x_new, np.sin(X*i*k)**2, axis=1)    
        return x_new
    
     # Plotting of Function with Datapoints
    for k in range(1, 11):
        plt.figure()
        for d in range(0, 7):
            x = get_coefficients(X, k=k, d=d)
            xt= x.T
            p=np.dot(xt,x)
            i=np.linalg.inv(p)
            xty=np.dot(xt,y)
            b=np.dot(i,xty)
            pred = np.dot(x, b)
            title = 'k={}'.format(k) , 'Training Data Size = 20'
            plt.title(title)
            plt.xlabel("X-value")
            plt.ylabel("Y-Prediction")
            plt.scatter(X,pred, label='d={}'.format(d),)
            plt.legend(loc = 'center left', bbox_to_anchor=(1, 0.5))
    plt.show()
    
    #Computing the Error on the Test Data
    for k in range(1, 11):
        print("\nFor k = ",k,'\n')
        plt.figure()
        for d in range(0, 7):
            x = get_coefficients(X, k=k, d=d)
            xt= x.T
            p=np.dot(xt,x)
            i=np.linalg.inv(p)
            xty=np.dot(xt,y)
            b=np.dot(i,xty)
            xtest2 = get_coefficients(Xtest, k=k, d=d)
            pred = np.dot(xtest2, b)
            error = np.sqrt(np.mean((pred - ytest)**2))
            print("Error for d =", d, " MSE = ", error)
            xlabel = 'k={}'.format(k)
            plt.title('Training Data Size = 20')
            plt.xlabel(xlabel)
            plt.ylabel("Mean Square Error")
            plt.scatter(k,error, label='d={}'.format(d),)
            plt.legend(loc = 'center left', bbox_to_anchor=(1, 0.5))
    plt.show()
            
    ### With 20 training data points, the error has increased and is now greater than 1 for some depths, indicating that the data is overfitted. Overfitting has increased as training data points have decreased.
    ### With all Training data points minimum error was 0.02506655544102931 for k=7 and d=4. Now minimum error has increased to 0.07206923426238365 for k=7 and d=6.
    
    print('\nEND Q1_D\n')


if __name__ == "__main__":
    main()


# In[ ]:




