import numpy as np
import math
import pandas as pd
import re
import nltk
from nltk.corpus import brown
from nltk.tokenize import word_tokenize
import pickle
import text_funcs as tf
import random
import os
from os.path import isfile, join
import time
from model_config import get_config
from scipy import stats

config = get_config()

class Word2Vec(object):
    def __init__(self, corpus = None, embedding_size = config["embedding_size"], epochs = config["epochs"], step_size = config["step_size"], optimizer = config["optimizer"], verbose = config["verbose"], cache = config["cache"]):
        # make sure optimizer is valid
        if optimizer not in config["valid_optimizers"]:
            raise ValueError("The optimizer that you input is not a valid. Please select a valid optimizer from the list below. \n" 
            + ", ".join(config["valid_optimizers"]))

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
        loss = -(1/N) * np.sum((Y*np.log(p+.001)) + ((1-Y)*np.log((1-p)+.001)))
        return loss

    def train(self, batch_size):
        """train the network"""

        # if chache exists, use those values
        files = [f for f in os.listdir(".") if isfile(join(".", f))]
        if self.architecture + "_cache.pkl" in files:
            print("recovering cached parameters")
            with open(self.architecture + "_cache.pkl", 'rb') as f:
                params = pickle.load(f)

            self.W1 = params["W1"]
            self.W2 = params["W2"]
            self.b1 = params["b1"]
            self.b2 = params["b2"]

        else:
            # create batches
            # random.seed(123)
            # inds = random.sample(list(range(self.X.shape[0])), self.X.shape[0])
            # self.X = self.X[inds]
            # self.Y = self.Y[inds]
            j = 1
            
            # start time
            t0 = time.time()

            # initialize Adam parameters
            if self.optimizer == "adam":
                a1 = .9
                a2 = .999
                m = [0, 0, 0, 0]
                v = [0, 0, 0, 0]

            print("start training")            

            # iterate over epochs
            for i in range(self.epochs):
                for k in range(0, self.X.shape[0], batch_size):
                    # get indices
                    temp_X = self.X[k:min(k+batch_size, self.X.shape[0])]
                    np.random.shuffle(temp_X)
                    temp_Y = self.Y[k:min(k+batch_size, self.X.shape[0])]
                    np.random.shuffle(temp_Y)

                    # do forward and back propigation on a given batch
                    p, h = self.forward_propigation(temp_X, W1 = self.W1, W2 = self.W2, b1 = self.b1, b2 = self.b2)
                    self.back_propigation(temp_X, temp_Y, p, h)
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
                            p_, _ = self.forward_propigation(temp_X, W1 = temp_W1, W2 = self.W2, b1 = self.b1, b2 = self.b2)
                            df = (-1/(temp_Y.shape[0])) * (temp_Y*(1-p_))
                            if self.get_loss(p_, temp_Y) > self.get_loss(p, temp_Y) + alpha*t*-(np.linalg.norm(np.matmul(temp_X.T,
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
                            p_, h_ = self.forward_propigation(temp_X, W1 = self.W1, W2 = temp_W2, b1 = self.b1, b2 = self.b2)
                            df = (-1/(temp_Y.shape[0])) * (temp_Y*(1-p_))
                            if self.get_loss(p_, temp_Y) > self.get_loss(p, temp_Y) + alpha*t*-(np.linalg.norm(np.matmul(h_.T, df))**2):
                                t = beta*t
                            else:
                                break
                        l_rates.append(t)
                        del temp_W2

                        t = .1
                        temp_b1 = self.b1.copy()
                        for k in range(3):
                            temp_b1 = temp_b1 - (t*self.db1)
                            p_, _ = self.forward_propigation(temp_X, W1 = self.W1, W2 = self.W2, b1 = temp_b1, b2 = self.b2)
                            df = (-1/(temp_Y.shape[0])) * (temp_Y*(1-p_))
                            if self.get_loss(p_, temp_Y) > self.get_loss(p, temp_Y) + alpha*t*-(np.linalg.norm(sum(np.matmul(df, self.W2.T)))**2):
                                t = beta*t
                            else:
                                break
                        l_rates.append(t)
                        del temp_b1

                        t = .1
                        temp_b2 = self.b2.copy()
                        for k in range(3):
                            temp_b2 = temp_b2 - (t*self.db2)
                            p_, _ = self.forward_propigation(temp_X, W1 = self.W1, W2 = self.W2, b1 = self.b1, b2 = temp_b2)
                            df = (-1/(temp_Y.shape[0])) * (temp_Y*(1-p_))
                            if self.get_loss(p_, temp_Y) > self.get_loss(p, temp_Y) + alpha*t*-(np.linalg.norm(sum(df)**2)):
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
                        print("iteration: " + str(j) + "/" + str(len(range(0, self.X.shape[0], batch_size))*self.epochs) + "\nloss: " + str(self.get_loss(p, temp_Y)))
                    j = j+1
            
            # end time
            t1 = time.time()

            if self.cache:
                # cache model parameters
                cache_data = {"W1": self.W1, "W2": self.W2, "b1": self.b1, "b2": self.b2}
                with open(self.architecture+"_cache.pkl", 'wb') as f:
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
        # return dict{word:embedding}
        return  dict(zip(list(self.lookup.values()), embeddings))

class ContinuousBagOfWords(Word2Vec):
    def __init__(self, corpus = None, embedding_size = 50, epochs = 3, step_size = .01, optimizer = "adam", verbose = True, cache = True):
        # run init method of paretn class Word2Vec
        super().__init__(corpus, embedding_size, epochs, step_size, optimizer, verbose, cache)

        # default corpus
        if corpus is None:
            tokens = brown.words(categories = "news")[:50000]
        else:
            tokens = word_tokenize(corpus)

        # get data
        self.X, self.Y, self.lookup = tf.get_input_output(tokens)

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

        self.architecture = "ContinuousBagOfWords"


class SkipGram(Word2Vec):
    def __init__(self, corpus = None, embedding_size = 50, epochs = 3, step_size = .01, optimizer = "adam", verbose = True, cache = True):
        # run init method of paretn class Word2Vec
        super().__init__(corpus, embedding_size, epochs, step_size, optimizer, verbose, cache)

        # default corpus
        if corpus is None:
            tokens = brown.words(categories = "news")[:50000]
        else:
            tokens = word_tokenize(corpus)

        # get data
        self.X, self.Y, self.lookup = tf.get_input_output(tokens, skip_gram = True)

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

        self.architecture = "SkipGram"

class SkipGram_NN(Word2Vec):
    def __init__(self, corpus = None, layers = None, step_size = config["step_size"], optimizer = "adam", verbose = True, cache = config["cache"]):
        # set up neural network architecture
        if layers is None:
            self.layers = [50]
        else:
            self.layers = layers

        epochs = config["epochs"]
        # run init method of parent class Word2Vec
        super().__init__(corpus, self.layers[-1], epochs, step_size, optimizer, verbose, cache)

        if corpus is None:
            tokens = brown.words(categories = "news")[:50000]
        else:
            tokens = word_tokenize(corpus)

        # get data
        self.X, self.Y, self.lookup = tf.get_input_output(tokens, skip_gram = True)

        # print size of vocab
        self.V = self.Y.shape[1]
        print("Vocab: " + str(self.V))

        # randomly initialize weights
        self.arch = [self.V] + self.layers + [self.V]
        self.W = []
        self.b = []
        for i in range(len(self.arch) - 1):
            random.seed(i+1)
            self.W.append(np.random.normal(0, .1, [self.arch[i], self.arch[i+1]]))
            self.b.append(np.zeros(self.arch[i+1]))
            
        # random.seed(1)
        # self.W1 = np.random.normal(0, .1, [self.V, self.embedding_size])
        # random.seed(2)
        # self.W2 = np.random.normal(0, .1, [self.embedding_size, self.V])

        # # Initialize biases to 0
        # self.b1 = np.zeros(self.embedding_size)
        # self.b2 = np.zeros(self.V)

        self.architecture = "SkipGram"

    def forward_propigation(self, X, W, b, hidden = False):
        """do a forward pass for through the model"""
        hidden_layers = []
        h = np.matmul(X, W[0]) + b[0]
        hidden_layers.append(h)

        for i in range(len(W)-2):
            h = np.matmul(h, W[i+1]) + b[i+1]
            hidden_layers.append(h)

        if hidden:
            # for the creation of the embeddings
            return hidden_layers
        else:
            f = np.matmul(h, W[-1]) + b[-1]
            f = f - np.max(f, axis=-1, keepdims=True)
            ef = np.exp(f)
            p = ef/ef.sum(axis=1)[:, None]
            # p[p<1e-10] = 0
            return p, hidden_layers

    def back_propigation(self, X, Y, p, hidden_layers):    
        """get the direction of step"""
        # get derivative of cross entropy
        # df = (-1/(Y.shape[0])) * (Y*(1-p))
        df = Y-p

        # initialize gradient lists
        self.dW = [None] * len(self.W)
        self.db = [None] * len(self.b)

        # set up 
        self.dW[-1] = np.matmul(hidden_layers[-1].T, df)
        self.db[-1] = sum(df)

        w = np.matmul(df, self.W[-1].T)
        for i in range(len(self.W)-2):
            if i == 0:
                pass
            else:
                w = np.matmul(w, self.W[-(i+1)].T)
            self.dW[-(i+2)] = np.matmul(hidden_layers[-(i+2)].T, w)
            # self.db[-(i+2)] = sum(np.matmul(df, w))
            self.db[-(i+2)] = sum(w)

        if len(self.W) > 2:
            w = np.matmul(w, self.W[1].T)
        self.dW[0] = np.matmul(X.T, w)
        self.db[0] = sum(w)

    def test(self):
        # do forward and back propigation on a given batch
        p, h = self.forward_propigation(X = self.X[:20], W = self.W, b = self.b)
        self.back_propigation(self.X[:20], self.Y[:20], p, h)
        assert [i.shape for i in self.W] == [i.shape for i in self.dW] 
        assert [i.shape for i in self.b] == [i.shape for i in self.db]
        self.update_parameters(1)
        print("Passed!")

    def update_parameters(self, l):
        """update parameters of model"""
        if isinstance(l, int) or isinstance(l, float):
            self.W = [self.W[i] - l*self.dW[i] for i in range(len(self.W))]
            self.b = [self.b[i] - l*self.db[i] for i in range(len(self.b))]
        else:
            self.W = [self.W[i] - l[i]*self.dW[i] for i in range(len(self.W))]
            self.b = [self.b[i] - l[len(self.b) + i]*self.db[i] for i in range(len(self.b))]        

    def train(self, batch_size):
        """train the network"""

        # if chache exists, use those values
        files = [f for f in os.listdir(".") if isfile(join(".", f))]
        if self.architecture + "_cache.pkl" in files and self.cache:
            print("recovering cached parameters")
            with open(self.architecture + "_cache.pkl", 'rb') as f:
                params = pickle.load(f)

            self.W = params["W"]
            self.b = params["b"]

        else:
            # create batches
            # random.seed(123)
            # inds = random.sample(list(range(self.X.shape[0])), self.X.shape[0])
            # self.X = self.X[inds]
            # self.Y = self.Y[inds]
            j = 1
            
            # start time
            t0 = time.time()

            # initialize Adam parameters
            if self.optimizer == "adam":
                a1 = .9
                a2 = .999
                m = [0] * (len(self.W) + len(self.b))
                v = [0] * (len(self.W) + len(self.b))

            print("start training")            

            # iterate over epochs
            for i in range(self.epochs):
                for k in range(0, self.X.shape[0], batch_size):
                    # get indices
                    temp_X = self.X[k:min(k+batch_size, self.X.shape[0])]
                    np.random.shuffle(temp_X)
                    temp_Y = self.Y[k:min(k+batch_size, self.X.shape[0])]
                    np.random.shuffle(temp_Y)

                    # do forward and back propigation on a given batch
                    p, h = self.forward_propigation(temp_X, W = self.W, b = self.b)
                    self.back_propigation(temp_X, temp_Y, p, h)
                    if self.optimizer == "gradient_descent_fixed":
                        self.update_parameters(self.step_size)
                    elif self.optimizer == "gradient_descent":
                        self.update_parameters(self.step_size/j)
                    elif self.optimizer == "adam":
                        # estimate first and second moments
                        for k in range(len(self.W)):
                            m[k] = a1 * m[k] + (1-a1) * self.dW[k]
                            v[k] = a2 * v[k]  + (1-a2) * (self.dW[k]*self.dW[k])
                            m_hat = m[k]/(1-(a1**(i+1)))
                            v_hat = v[k]/(1-(a2**(i+1)))

                            # move weight by mean/ standard devations
                            self.W[k] = self.W[k] - self.step_size*m_hat/((v_hat**.5)+1e-8)

                        # estimate first and second moments
                        for k in range(len(self.b)):
                            m[len(self.W) + k] = a1 * m[len(self.W) + k]  + (1-a1) * self.b[k]
                            v[len(self.W) + k] = a2 * v[len(self.W) + k]  + (1-a2) * (self.b[k]*self.b[k])
                            m_hat = m[len(self.W) + k]/(1-(a1**(i+1)))
                            v_hat = v[len(self.W) + k]/(1-(a2**(i+1)))
                            
                            # move weight by mean/ standard devations
                            self.b[k] = self.b[k] - self.step_size*m_hat/((v_hat**.5)+1e-8)

                    # print loss
                    if j%50 == 0 and self.verbose:
                        print("iteration: " + str(j) + "/" + str(len(range(0, self.X.shape[0], batch_size))*self.epochs) + "\nloss: " + str(self.get_loss(p, temp_Y)))
                    j = j+1
            
            # end time
            t1 = time.time()

            if self.cache:
                # cache model parameters
                cache_data = {"W": self.W, "b": self.b}
                with open(self.architecture+"_cache.pkl", 'wb') as f:
                    pickle.dump(cache_data, f)

            # print run time
            print("Training Complete\nRun Time: " + str(t1-t0) + " seconds")

    def create_embeddings(self):
        # embeddings = .5*self.W[0] + .5*self.W[-1].T
        embeddings = .5*(self.W[0] + self.b[0]) + .5*(self.W[-1] + self.b[-1]).T
        print(self.W[1])
        print(self.b[1])

        embedding_list = []
        embeddings_df = pd.DataFrame(embeddings)
        embeddings_df.insert(loc = 0, column = "word", value = list(self.lookup.values()))
        for i in range(embeddings_df.shape[0]):
            if embeddings_df["word"][i] != "bath":
                dot=np.dot(np.array(embeddings_df.iloc[i, 1:]), np.array(embeddings_df.iloc[6286, 1:]))
                embedding_list.append(dot/(np.linalg.norm(np.array(embeddings_df.iloc[i, 1:])) * np.linalg.norm(np.array(embeddings_df.iloc[6286, 1:]))))
        print(min(np.array(embedding_list)))
        print(np.quantile(np.array(embedding_list), .1))
        print(np.quantile(np.array(embedding_list), .2))
        print(np.quantile(np.array(embedding_list), .3))
        print(np.quantile(np.array(embedding_list), .4))
        print(np.quantile(np.array(embedding_list), .5))
        print(np.quantile(np.array(embedding_list), .7))
        print(np.quantile(np.array(embedding_list), .8))
        print(np.quantile(np.array(embedding_list), .9))
        print(max(np.array(embedding_list)))
        print(embeddings_df.head())

        embeddings_df.to_csv("test.csv", index = False)
        # return dict{word:embedding}
        return embeddings_df

    def embed(self, query):
        word_to_id = {v: k for k, v in self.lookup.items()}
        X = tf.embed_text(query, word_to_id)
        embedding = self.forward_propigation(X = X, W = self.W, b = self.b, hidden = True)
        for i in range(len(embedding)):
            embedding[i] = embedding[i]/np.std(embedding[i])
        output = sum(embedding)/len(embedding)
        print(X)
        return output
        
if __name__ == "__main__":
    # W2V = SkipGram(step_size = .001, optimizer = "adam")
    # W2V.train(batch_size = config["batch_size"])
    # embeddings = W2V.create_embeddings()
    # with open(W2V.architecture + ".pkl", 'wb') as f:
    #     pickle.dump(embeddings, f)

    # embedding_list = []
    # for key,val in embeddings.items():
    #     if key != "bath":
    #         dot=np.dot(embeddings["bath"], embeddings[key])
    #         embedding_list.append(dot/(np.linalg.norm(embeddings["bath"])*np.linalg.norm(embeddings[key])))
    # print(max(embedding_list))
    # print(min(embedding_list))
    W2V = SkipGram_NN(layers = [100, 100, 100], step_size = .01, optimizer = "adam", cache = config["cache"])
    W2V.train(batch_size = config["batch_size"])
    W2V.create_embeddings()
    print(W2V.embed("The county of the lake"))
