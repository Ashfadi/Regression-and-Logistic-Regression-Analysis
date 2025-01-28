#!/usr/bin/env python
# coding: utf-8

# In[15]:


def main():
    print('START Q1_AB\n')
    
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
    
    #Assigning X,y values for Training Data
    X = Training_Data[:,[0]]
    y = Training_Data[:,[1]]
    
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
            title = 'k={}'.format(k) , 'Training Data Size = 128'
            plt.title(title)
            plt.xlabel("X-value")
            plt.ylabel("Y-Prediction")
            plt.scatter(X,pred, label='d={}'.format(d),)
            plt.legend(loc = 'center left', bbox_to_anchor=(1, 0.5))
    plt.show()
    
    print('\nEND Q1_AB\n')


if __name__ == "__main__":
    main()


# In[ ]:




