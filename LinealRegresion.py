import matplotlib.pyplot as ptl
import pandas as pd
import numpy as np

# Clase de regresión lineal
class Regresion:
    def __init__(self, dim):
        self.dim = dim
        self.m_W = np.random.rand(dim)
        self.m_W  = self.m_W.reshape(-1, 1)
        self.m_b = np.random.random()

    def H(self, X):
        return np.dot(X, self.m_W) + self.m_b

    def Loss(self, X, Y):
        y_pred = self.H(X)
        return (np.linalg.norm((Y - y_pred))**2)/(2*len(Y)), y_pred
    def Loss_L2(self, X, Y, lambda_):
        y_pred = self.H(X)
        return (np.linalg.norm((Y - y_pred))**2)/(2*len(Y)), y_pred + lambda_*np.linalg.norm(self.m_W, 2)

    def Loss_L1(self, X, Y, lambda_):
        y_pred = self.H(X)
        return (np.linalg.norm((Y - y_pred))**2) / (2 * len(Y)), y_pred + lambda_ * np.linalg.norm(self.m_W, 1)

    def dL(self, X, Y, Y_pre):
        error = Y - Y_pre
        dw = -np.dot(X.T, error)/len(Y)
        db = np.sum((error)*(-1))/len(Y)
        return dw, db

    def change_params(self, dw, db, alpha):
        self.m_W = self.m_W - alpha*dw
        self.m_b = self.m_b - alpha*db

    def train(self, X, Y, alpha, epochs, _lambda, reg=""):
        error_list = []
        time_stamp = []
        for i in range(epochs):
            if reg == "L2":
                loss, y_pred = self.Loss_L2(X, Y, _lambda)
            elif reg == "L1":
                loss, y_pred = self.Loss_L1(X, Y, _lambda)
            else:
                loss, y_pred = self.Loss(X, Y)
            time_stamp.append(i)
            error_list.append(loss)
            dw, db = self.dL(X, Y, y_pred)
            self.change_params(dw, db, alpha)
            # print("error de pérdida : " + str(loss))
            if (i % 100 == 0):
                print("error de pérdida : " + str(loss))
                #    self.plot_error(time_stamp, error_list)
                # self.plot_plane(X[:, 0], X[:, 1], Y)

        return time_stamp, error_list

    def plot_error(self, time, loss):
        ptl.plot(time, loss)
        ptl.show()

    def plot_line(self, x, y_pre):
        ptl.plot(x, y_pre, '*')
        ptl.plot(x, self.H(x))
        ptl.show()

