#!/usr/bin/env python
# coding: utf-8

# In[1]:


def main():
    print('START Q3_C\n')
    
    # Importing Libraries and Dataset
    import numpy as np
    from random import randrange
    
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
    
    # Finding the minimum and maximum values for each column
    def dataset_minmax(dataset):
        minmax = list()
        for i in range(len(dataset[0])):
            col_values = [row[i] for row in dataset]
            value_min = min(col_values)
            value_max = max(col_values)
            minmax.append([value_min, value_max])
        return minmax

    # Rescale Training data columns to the range 0-1
    def normalize_dataset(dataset, minmax):
        for row in dataset:
            for i in range(len(row)):
                row[i] = (row[i] - minmax[i][0]) / (minmax[i][1] - minmax[i][0])

    # Split Training Data into k folds
    def cross_validation_split(dataset, n_folds):
        dataset_split = list()
        dataset_copy = list(dataset)
        fold_size = int(len(dataset) / n_folds)
        for i in range(n_folds):
            fold = list()
            while len(fold) < fold_size:
                index = randrange(len(dataset_copy))
                fold.append(dataset_copy.pop(index))
            dataset_split.append(fold)
        return dataset_split

    # Calculate accuracy percentage
    def accuracy_metric(actual, predicted):
        correct = 0
        for i in range(len(actual)):
            if actual[i] == predicted[i]:
                correct += 1
        return correct / float(len(actual)) * 100.0

    # Evaluate an algorithm using a cross validation split
    def evaluate_algorithm(dataset, algorithm, n_folds, *args):
        folds = cross_validation_split(dataset, n_folds)
        scores = list()
        for fold in folds:
            train_set = list(folds)
            def remove_values_from_list(train_set, fold):
                return [value for value in train_set if value != fold]
            train_set = sum(train_set, [])
            test_set = list()
            for row in fold:
                row_copy = list(row)
                test_set.append(row_copy)
                row_copy[-1] = None
            predicted = algorithm(train_set, test_set, *args)
            actual = [row[-1] for row in fold]
            accuracy = accuracy_metric(actual, predicted)
            scores.append(accuracy)
        return scores

    # Make a prediction with coefficients
    def predict(row, coefficients):
        yhat = coefficients[0]
        for i in range(len(row)-1):
            yhat += coefficients[i + 1] * row[i]
        return 1.0 / (1.0 + np.exp(-yhat))

    # Estimate logistic regression coefficients using stochastic gradient descent
    def coefficients_sgd(train, lr, itr):
        coef = [0.0 for i in range(len(train[0]))]
        for iterations in range(itr):
            for row in train:
                yhat = predict(row, coef)
                error = row[-1] - yhat
                coef[0] = coef[0] + lr * error * yhat * (1.0 - yhat)
                for i in range(len(row)-1):
                    coef[i + 1] = coef[i + 1] + lr * error * yhat * (1.0 - yhat) * row[i]
        return coef

    # Linear Regression Algorithm With Stochastic Gradient Descent
    def logistic_regression(train, test, lr, itr):
        predictions = list()
        coef = coefficients_sgd(train, lr, itr)
        for row in test:
            yhat = predict(row, coef)
            yhat = round(yhat)
            predictions.append(yhat)
        return(predictions)

    minmax = dataset_minmax(Training_Data)
    normalize_dataset(Training_Data, minmax)

    # evaluate algorithm
    scores = evaluate_algorithm(Training_Data, logistic_regression, 120, 0.01, 80)
    accuracy=(sum(scores)/float(len(scores)))
    print('Height, Weight, Age','\nFor alpha = 0.01 , iterations = 80','\nLeave one out Accuracy = %.3f%%' % accuracy)
    
    # The accuracy of prediction using Logistic Regression increases as the iteration is increasing but it take time to run as well.
    # I obtained the same accuracy as the Nave Bayes Classifier after 5 iterations which is 55%.
    # I obtained the same accuracy as the KNN Classifier when K=11 after 70 iterations which is 68.33%.
    # I got 69.167% accuracy after 80 iterations. It drops to 68.33% after 100 iterations.
    # I achieved 70.833% accuracy after 400 iterations to 1000 iterations. After 1000 iterations, the accuracy remains constant at 71.667%.
    # KNN Model and Logistic Regression outperforms Gaussian Naïve Bayes Classifier in predicting gender for the given dataset. 
    # The accuracy of the KNN Model is greater than that of the Logistic Regression.
    # The accuracy of Logistic Regression is greater than that of Gaussian Naïve Bayes Classifier.
    
    print('\nEND Q3_C\n')


if __name__ == "__main__":
    main()


# In[ ]:




