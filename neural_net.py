import numpy as np
import math
import re
import nltk
from nltk.corpus import brown
import pickle
import text_funcs as tf
import random
import os
from os.path import isfile, join
import time

class Word2Vec(object):
    def __init__(self, embedding_size = 50, epochs = 3, step_size = .01, optimizer = "gradient_descent_fixed", verbose = True, cache = True):
        # get data
        self.X, self.Y, self.lookup = tf.get_input_output()

        # determine size of embedding
        self.embedding_size = embedding_size

        # set hyper parameters
        self.epochs = epochs
        self.step_size = step_size

        # set optimizer
        self.optimizer = optimizer

        # if output/cache is wanted
        self.verbose = verbose
        self.cache = cache

        # print size of vocab
        self.V = self.Y.shape[1]
        print("Vocab: " + str(self.V))

        # randomly initialize weights
        random.seed(1)
        self.W1 = np.random.normal(0, .1, [self.V, self.embedding_size])
        random.seed(2)
        self.W2 = np.random.normal(0, .1, [self.embedding_size, self.V])

        # Initialize biases to 0
        self.b1 = np.zeros(self.embedding_size)
        self.b2 = np.zeros(self.V)

    def forward_propigation(self, X, W1, W2, b1, b2, hidden = False):
        """do a forward pass for through the model"""
        h = np.matmul(X, W1) + b1
        if hidden:
            # for the creation of the embeddings
            return h
        else:
            f = np.matmul(h, W2) + b2
            f = f - np.max(f, axis=-1, keepdims=True)
            ef = np.exp(f)
            p = ef/ef.sum(axis=1)[:, None]
            # p[p<1e-10] = 0
            return p, h

    def back_propigation(self, X, Y, p, h):    
        """get the direction of step"""
        df = (-1/(Y.shape[0])) * (Y*(1-p))
        self.dW2 = np.matmul(h.T, df)
        self.db2 = sum(df)
        self.dW1 = np.matmul(X.T, np.matmul(df, self.W2.T))
        self.db1 = sum(np.matmul(df, self.W2.T))

    def update_parameters(self, l):
        """update parameters of model"""
        if isinstance(l, int) or isinstance(l, float):
            self.W1 = self.W1 - (l*self.dW1)
            self.W2 = self.W2 - (l*self.dW2)
            self.b1 = self.b1 - (l*self.db1)
            self.b2 = self.b2 - (l*self.db2)
        else:
            self.W1 = self.W1 - (l[0]*self.dW1)
            self.W2 = self.W2 - (l[1]*self.dW2)
            self.b1 = self.b1 - (l[2]*self.db1)
            self.b2 = self.b2 - (l[3]*self.db2)           

    def get_loss(self, p, Y):
        """print out loss"""
        N = p.shape[1]
        loss = -(1/N) * np.sum(np.sum((Y * np.log(p+.001)) + (Y * np.log(p+.001)), axis=0, keepdims=True), axis=1)
        return loss[0]

    def train(self, batch_size):
        """train the network"""

        # if chache exists, use those values
        files = [f for f in os.listdir(".") if isfile(join(".", f))]
        if self.optimizer+"_cache.pkl" in files:
            print("recovering cached parameters")
            with open(self.optimizer+"_cache.pkl", 'rb') as f:
                params = pickle.load(f)

            self.W1 = params["W1"]
            self.W2 = params["W2"]
            self.b1 = params["b1"]
            self.b2 = params["b2"]

        else:
            print("starting training")

            # create batches
            random.seed(123)
            inds = random.sample(list(range(self.X.shape[0])), self.X.shape[0])
            inds = [[inds[i:i+batch_size]] for i in range(0, self.X.shape[0], batch_size)]
            j = 1
            
            # start time
            t0 = time.time()

            # initialize Adam parameters
            a1 = .9
            a2 = .999
            m = [0, 0, 0, 0]
            v = [0, 0, 0, 0]

            # iterate over epochs
            for i in range(self.epochs):
                for batch in inds:
                    # do forward and back propigation on a given batch
                    p, h = self.forward_propigation(self.X[batch], W1 = self.W1, W2 = self.W2, b1 = self.b1, b2 = self.b2)
                    self.back_propigation(self.X[batch], self.Y[batch], p, h)
                    if self.optimizer == "gradient_descent_fixed":
                        self.update_parameters(self.step_size)
                    elif self.optimizer == "gradient_descent":
                        self.update_parameters(self.step_size/j)
                    elif self.optimizer == "gradient_descent_backtracking_line_search":
                        # initialize parameters for line search
                        random.seed(j)
                        alpha = random.random()*.5
                        random.seed(self.V*batch_size + j)
                        beta = random.uniform(.5, 1) 
                        
                        # perform backtracking line search for each variable individually
                        l_rates=[]
                        t = .1
                        temp_W1 = self.W1.copy()
                        for k in range(3):
                            temp_W1 = temp_W1 - (t*self.dW1)
                            p_, _ = self.forward_propigation(self.X[batch], W1 = temp_W1, W2 = self.W2, b1 = self.b1, b2 = self.b2)
                            df = (-1/(self.Y[batch].shape[0])) * (self.Y[batch]*(1-p_))
                            if self.get_loss(p_, self.Y[batch]) > self.get_loss(p, self.Y[batch]) + alpha*t*-(np.linalg.norm(np.matmul(self.X[batch].T,
                            np.matmul(df, self.W2.T)))**2):
                                t = beta*t
                            else:
                                break
                        l_rates.append(t)
                        del temp_W1

                        t = .1
                        temp_W2 = self.W2.copy()
                        for k in range(3):
                            temp_W2 = temp_W2 - (t*self.dW2)
                            p_, h_ = self.forward_propigation(self.X[batch], W1 = self.W1, W2 = temp_W2, b1 = self.b1, b2 = self.b2)
                            df = (-1/(self.Y[batch].shape[0])) * (self.Y[batch]*(1-p_))
                            if self.get_loss(p_, self.Y[batch]) > self.get_loss(p, self.Y[batch]) + alpha*t*-(np.linalg.norm(np.matmul(h_.T, df))**2):
                                t = beta*t
                            else:
                                break
                        l_rates.append(t)
                        del temp_W2

                        t = .1
                        temp_b1 = self.b1.copy()
                        for k in range(3):
                            temp_b1 = temp_b1 - (t*self.db1)
                            p_, _ = self.forward_propigation(self.X[batch], W1 = self.W1, W2 = self.W2, b1 = temp_b1, b2 = self.b2)
                            df = (-1/(self.Y[batch].shape[0])) * (self.Y[batch]*(1-p_))
                            if self.get_loss(p_, self.Y[batch]) > self.get_loss(p, self.Y[batch]) + alpha*t*-(np.linalg.norm(sum(np.matmul(df, self.W2.T)))**2):
                                t = beta*t
                            else:
                                break
                        l_rates.append(t)
                        del temp_b1

                        t = .1
                        temp_b2 = self.b2.copy()
                        for k in range(3):
                            temp_b2 = temp_b2 - (t*self.db2)
                            p_, _ = self.forward_propigation(self.X[batch], W1 = self.W1, W2 = self.W2, b1 = self.b1, b2 = temp_b2)
                            df = (-1/(self.Y[batch].shape[0])) * (self.Y[batch]*(1-p_))
                            if self.get_loss(p_, self.Y[batch]) > self.get_loss(p, self.Y[batch]) + alpha*t*-(np.linalg.norm(sum(df)**2)):
                                t = beta*t
                            else:
                                break
                        l_rates.append(t)
                        del temp_b2

                        self.update_parameters(l_rates)

                    elif self.optimizer == "adam":
                        # estimate first and second moments
                        m[0] = a1 * m[0] + (1-a1) * self.dW1
                        v[0] = a2 * v[0]  + (1-a2) * (self.dW1*self.dW1)
                        m_hat = m[0]/(1-(a1**(i+1)))
                        v_hat = v[0]/(1-(a2**(i+1)))

                        # move weight by mean/ standard devations
                        self.W1 = self.W1 - self.step_size*m_hat/((v_hat**.5)+1e-8)

                        m[1] = a1 * m[1]  + (1-a1) * self.dW2
                        v[1] = a2 * v[1]  + (1-a2) * (self.dW2*self.dW2)
                        m_hat = m[1]/(1-(a1**(i+1)))
                        v_hat = v[1]/(1-(a2**(i+1)))
                        self.W2 = self.W2 - self.step_size*m_hat/((v_hat**.5)+1e-8)

                        m[2] = a1 * m[2]  + (1-a1) * self.b1
                        v[2] = a2 * v[2]  + (1-a2) * (self.b1*self.b1)
                        m_hat = m[2]/(1-(a1**(i+1)))
                        v_hat = v[2]/(1-(a2**(i+1)))
                        self.b1 = self.b1 - self.step_size*m_hat/((v_hat**.5)+1e-8)

                        m[3] = a1 * m[3] + (1-a1) * self.db2
                        v[3] = a2 * v[3]  + (1-a2) * (self.db2*self.db2)
                        m_hat = m[3]/(1-(a1**(i+1)))
                        v_hat = v[3]/(1-(a2**(i+1)))
                        self.b2 = self.b2 - self.step_size*m_hat/((v_hat**.5)+1e-8)

                    # print loss
                    if j%50 == 0 and self.verbose:
                        print("iteration: " + str(j) + "/" + str(len(inds)*self.epochs) + "\nloss: " + str(self.get_loss(p, self.Y[batch])))
                    j = j+1
            
            # end time
            t1 = time.time()

            if self.cache:
                # cache model parameters
                cache_data = {"W1": self.W1, "W2": self.W2, "b1": self.b1, "b2": self.b2}
                with open(W2V.optimizer+"_cache.pkl", 'wb') as f:
                    pickle.dump(cache_data, f)

            # print run time
            print("Training Complete\nRun Time: " + str(t1-t0) + " seconds")

    def create_embeddings(self):
        # intialize list for embeddings
        embeddings = list()
        for key, val in self.lookup.items():

            #affirm correct input
            X = np.zeros(self.V)
            X[key] = 1

            # get hidden layer value
            h = self.forward_propigation(X, W1 = self.W1, W2 = self.W2, b1 = self.b1, b2 = self.b2, hidden=True)
            embeddings.append(h)
        #return dict{word:embedding}
        return  dict(zip(list(self.lookup.values()), embeddings))

if __name__ == "__main__":
    W2V = Word2Vec(step_size = .001, optimizer = "adam")
    W2V.train(batch_size = 100)
    embeddings = W2V.create_embeddings()
    with open(W2V.optimizer+".pkl", 'wb') as f:
        pickle.dump(embeddings, f)

    embedding_list = []
    for key,val in embeddings.items():
        dot=np.dot(embeddings["bath"], embeddings[key])
        embedding_list.append(dot/(np.linalg.norm(embeddings["bath"])*np.linalg.norm(embeddings[key])))
    print(max(embedding_list))
    print(min(embedding_list))