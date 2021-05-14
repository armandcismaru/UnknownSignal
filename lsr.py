import os
import random
import sys
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

def load_points_from_file(filename):
    points = pd.read_csv(filename, header=None)
    return points[0].values, points[1].values

def least_squares(xi, yi): 
    xi = xi.reshape((xi.size, 1))
    yi = yi.reshape((yi.size, 1))
    X = np.column_stack((np.ones(xi.shape), xi))
    A = np.linalg.inv(np.dot(np.transpose(X), X)).dot(np.transpose(X)).dot(yi)
    return A[:,0]

def least_squares_poly(xi, yi): 
    X = np.column_stack((np.ones(xi.shape), xi, xi ** 2, xi ** 3))
    A = np.linalg.inv(np.dot(np.transpose(X), X)).dot(np.transpose(X)).dot(yi)
    return A

def least_squares_sin(xi, yi): 
    xi = xi.reshape((xi.size, 1))
    yi = yi.reshape((yi.size, 1))
    X = np.column_stack((np.ones(xi.shape), np.sin(xi)))
    A = np.linalg.inv(np.dot(np.transpose(X), X)).dot(np.transpose(X)).dot(yi)
    return A[:,0]

def square_error(y, y_hat):
    return np.sum((y - y_hat) ** 2)
    
def plot_linear(xs, ys):
    a, b = least_squares(xs, ys)
    y1 = a + b * xs.min()
    y2 = a + b * xs.max()
    plt.plot([xs.min(), xs.max()], [y1, y2], 'r-', lw=2)   

def plot_polynomial(xs, ys):
    a, b, c, d = least_squares_poly(xs, ys)
    new_xs = np.linspace(xs.min(), xs.max(), 20)
    new_ys = a + b * new_xs + c * (new_xs ** 2) + d * (new_xs ** 3) 
    plt.plot(new_xs, new_ys, c='r')

def plot_unknown(xs, ys):
    a, b = least_squares_sin(xs, ys)
    new_xs = np.linspace(xs.min(), xs.max(), 20)
    new_ys = a + b * np.sin(new_xs)
    plt.plot(new_xs, new_ys, c='r')

def poly_test(x_train, y_train, x_test, y_test):
    a, b, c, d = least_squares_poly(x_train, y_train)
    yh_test = a + b * x_test + c * (x_test ** 2) + d * (x_test ** 3) 
    return np.mean((y_test - yh_test)**2)

def linear_test(x_train, y_train, x_test, y_test):
    a, b = least_squares(x_train, y_train)
    yh_test = a + b * x_test
    return np.mean((y_test - yh_test)**2)

def unknown_test(x_train, y_train, x_test, y_test):
    a, b = least_squares_sin(x_train, y_train)
    yh_test = a + b * np.sin(x_test)
    return np.mean((y_test - yh_test)**2)

def cross_validation(xs, ys):
    random.seed(1)
    mapIndexPosition = list(zip(xs, ys))
    random.shuffle(mapIndexPosition)
    xs, ys = zip(*mapIndexPosition)

    k_number = 10
    kfold_x = np.array_split(xs, k_number)
    kfold_y = np.array_split(ys, k_number)
    med_lin = med_sin = med_poly = 0
    
    for i in range(k_number):
        x_test = kfold_x[i]
        y_test = kfold_y[i]
        x_train = y_train = []
        for k in range(len(kfold_x)):
            if i != k: 
                x_train = np.concatenate((x_train,kfold_x[k]))
                y_train = np.concatenate((y_train,kfold_y[k]))
        med_lin  += linear_test(x_train, y_train, x_test, y_test)
        med_poly += poly_test(x_train, y_train, x_test, y_test)
        med_sin  += unknown_test(x_train, y_train, x_test, y_test)
    
    if min(med_sin/k_number, med_poly/k_number, med_lin/k_number) == med_poly/k_number: type = 'Polynomial'
    elif min(med_sin/k_number, med_poly/k_number, med_lin/k_number) == med_sin/k_number: type = 'Unknown'
    else: type = 'Linear'  
    return type

def get_error(xs, ys, type):
    if type == 'Polynomial':
        a, b, c, d = least_squares_poly(xs, ys)
        y_hat_poly = a + b * xs + c * (xs ** 2) + d * (xs ** 3)
        error = square_error(ys, y_hat_poly)

    if type == 'Linear':
        a, b = least_squares(xs, ys)
        y_hat = a + b * xs
        error = square_error(ys, y_hat)
    
    if type == 'Unknown':
        a, b = least_squares_sin(xs, ys)
        y_hat_sin = a + b * np.sin(xs)
        error = square_error(ys, y_hat_sin)
    return error

def view_data_segments(xs, ys, plot):
    assert len(xs) == len(ys)
    assert len(xs) % 20 == 0
    reconstruction_error = 0
    len_data = len(xs)
    num_segments = len_data // 20
    
    if plot == True:
        colour = np.concatenate([[i] * 20 for i in range(num_segments)])
        plt.set_cmap('Dark2')
        plt.scatter(xs, ys, c=colour)
    
    for i in range (num_segments):
        type = cross_validation(xs[i*20:(i+1)*20], ys[i*20:(i+1)*20])
        if plot == True:
            if type == 'Linear':
                plot_linear(xs[i*20:(i+1)*20], ys[i*20:(i+1)*20])
            elif type == 'Polynomial': 
               plot_polynomial(xs[i*20:(i+1)*20], ys[i*20:(i+1)*20])
            elif type == 'Unknown': 
                plot_unknown(xs[i*20:(i+1)*20], ys[i*20:(i+1)*20])
        reconstruction_error += get_error(xs[i*20:(i+1)*20], ys[i*20:(i+1)*20], type)
        
    print("Total reconstructon error is:", float(reconstruction_error))
    if plot: plt.show() 

if __name__ == "__main__":
    if len(sys.argv) > 2 and str(sys.argv[2]) == '--plot':
        plot = True
    else: plot = False
    p = os.path.abspath('datafiles/train_data/' + str(sys.argv[1]))
    file_to_open = open(p)
    xs, ys = load_points_from_file(open(p))
    view_data_segments(xs, ys, plot)
