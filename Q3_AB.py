#!/usr/bin/env python
# coding: utf-8

# In[17]:


def main():
    print('START Q3_AB\n')
    
    # Importing Libraries and Dataset
    import numpy as np
    import matplotlib.pyplot as plt
    get_ipython().run_line_magic('matplotlib', 'notebook')
    from mpl_toolkits.mplot3d import Axes3D
    
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

    training = './datasets/Q3_data.txt'
    Training_Data = readFile(training)
    for i in Training_Data:
        if i[3]=='W':
            i[3]=i[3].replace('W','1')
            i[3]=int(i[3])
        else:
            i[3]=i[3].replace('M','0')
            i[3]=int(i[3])
    Training_Data=Training_Data.astype(float)
    
    #Assigning Feature and target columns to X and y respectively
    X=Training_Data[:,[0,1,2]]
    y=Training_Data[:,[3]]
    
    #Making Logistic Regression Model
    class LogisticRegression():
        def __init__(self, learning_rate, iterations):        
            self.learning_rate = learning_rate        
            self.iterations = iterations
        #Function for model training    
        def fit(self, X, Y):        
            #no_of_training_examples, no_of_features        
            self.m, self.n = X.shape        
            # weight initialization        
            self.W = np.zeros(self.n)        
            self.b = 0        
            self.X = X        
            self.Y = Y
            # gradient descent learning
            for i in range(self.iterations):            
                self.update_weights()            
            return self
        #Function to update weights in gradient descent
        def update_weights(self):           
            A = 1 / ( 1 + np.exp( - ( self.X.dot( self.W ) + self.b ) ) )
            # calculate gradients        
            tmp = ( A - self.Y.T )        
            tmp = np.reshape( tmp, self.m )        
            dW = np.dot( self.X.T, tmp ) / self.m         
            db = np.sum( tmp ) / self.m 
            # update weights    
            self.W = self.W - self.learning_rate * dW    
            self.b = self.b - self.learning_rate * db
            return self
        # Hypothetical function  h(x) 
        def predict(self, X):    
            Z = 1 / ( 1 + np.exp( - ( X.dot( self.W ) + self.b ) ) )        
            Y = np.where( Z > 0.5, 1, 0 )        
            return Y
        #Plotting Decision Boundary in 3D
        def plot_decision_region_3d(self, X, y, model, resolution=0.02):
            # plot the decision surface
            x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
            x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1

            c = -model.b / model.W[2]
            a = -model.W[0] / model.W[2]
            b = -model.W[1] / model.W[2]
            xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                                        np.arange(x2_min, x2_max, resolution))
            Z = a * xx1 + b * xx2 + c
            fig = plt.figure(figsize=(6,6)).add_subplot(111, projection = '3d')
            ax = fig.plot_surface(xx1, xx2, Z, alpha=0.2)
            # plot class samples
            M = Training_Data[Training_Data[:,3] == 0]
            W = Training_Data[Training_Data[:,3] == 1]
            fig.scatter(M[:,[0]], M[:,[1]], M[:,[2]], linewidths=5, color = 'red', label = 'Man')
            fig.scatter(W[:,[0]], W[:,[1]], W[:,[2]], linewidth = 5, color = 'blue', label = 'Woman')
            fig.set_xlabel('Height')
            fig.set_ylabel('Weight')
            fig.set_zlabel('Age')
            fig.legend()

    model = LogisticRegression(learning_rate = 0.01, iterations = 1000)      
    model.fit(X, y) 
    model.plot_decision_region_3d(X, y, model)
    
    #Function to calculate accuracy of actual and predicted label
    def accuracy_metric(actual, predicted):
        correct = 0
        for i in range(len(actual)):
            if actual[i] == predicted[i]:
                correct += 1
        return correct / float(len(actual)) * 100.0
    
    #Computing Accuracy at each iteration
    for i in range (990,1001):
        model = LogisticRegression(learning_rate = 0.01, iterations=i)      
        model.fit(X, y) 
        y_pred=model.predict(X)
        accuracy = accuracy_metric(y, y_pred)
        print('Itr =', i, 'accuracy =',accuracy)
    
    print('\nEND Q3_AB\n')


if __name__ == "__main__":
    main()


# In[ ]:




