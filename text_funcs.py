import numpy as np 
import re
import nltk
from nltk.corpus import brown
from nltk.tokenize import word_tokenize
import pickle
from collections import Counter
import boto3

def call_lambda(FunctionName = "getTextData"):
    client = boto3.client('lambda')
    response = client.invoke(FunctionName = FunctionName)
    return response["Payload"].read()

def embed_text(text, word_to_id):
    vocab_size = len(word_to_id)

    # tokenize text
    tokens = word_tokenize(text)

    # clean the tokens to make sure they are appropriate for the modeling
    tokens = clean_text(tokens)

    # coerce all words to lowercase
    tokens = [token.lower() for token in tokens]

    # get indices
    inds = [word_to_id[token] for token in tokens if token in list(word_to_id.keys())]
    inds = dict(Counter(inds))

    # return vector of hot encoded input
    X = np.zeros(vocab_size)
    X[np.array(list(inds.keys()))] = np.array(list(inds.values()))/len(tokens)
    return X

def clean_text(words):
    """Remove non-ASCII characters from list of tokenized words"""
    new_words = []
    for word in words:
        new_word = re.sub(r'[^\w\s]', '', word)
        new_word = re.sub(r'\d', '', new_word)
        if new_word != '':
            new_words.append(new_word)
    return new_words

def get_mappings(tokens):
    """create a mapping for all words in the vocabulary"""
    word_to_id = dict()
    id_to_word = dict()
    j=0

    for i in range(len(tokens)):
        if tokens[i] not in list(word_to_id.keys()):
            word_to_id[tokens[i]] = j
            id_to_word[j] = tokens[i]
            j=j+1
    return word_to_id, id_to_word

def get_corpus_data(tokens, word_to_id, window_size, skip_gram = False):
    N = len(tokens)
    X, Y = [], []

    # iterate through each context
    for i in range(N):
        nbr_inds = list(range(max(0, i - window_size), i)) + list(range(i + 1, min(N, i + window_size + 1)))
        if skip_gram:
            # add the context to the input X
            context = [word_to_id[tokens[j]] for j in nbr_inds]

            # change the data cleaning for skip gram vs CBOW
            for c in context:
                X.append(c)
                Y.append(word_to_id[tokens[i]])

        else:            
            if len(nbr_inds) == 2*window_size:
                # add the context to the input X
                X.append([word_to_id[tokens[j]] for j in nbr_inds])
                Y.append(word_to_id[tokens[i]])
      
    X = np.stack(X, axis = 0)
    Y = np.array(Y)
    Y = np.expand_dims(Y, axis = 0)
    return X, Y

def get_input_output(tokens, skip_gram = False):
    # clean the tokens to make sure they are appropriate for the modeling
    tokens = clean_text(tokens)

    # coerce all words to lowercase
    tokens = [token.lower() for token in tokens]

    # generate the mappings
    word_to_id, id_to_word = get_mappings(tokens)
    X, Y = get_corpus_data(tokens, word_to_id, 2, skip_gram)
    vocab_size = len(id_to_word)
    m = Y.shape[1]
    
    # convert the word indices into the proper one-hot encoded input
    Y_2 = np.zeros((vocab_size, m))
    Y_2[Y.flatten(), np.arange(m)] = 1
    Y_2 = Y_2.T
    X_2 = np.zeros([vocab_size, m])
    if skip_gram:
        X_2[X.flatten(), np.arange(m)] = 1
        X_2 = X_2.T

        return Y_2, X_2, id_to_word

    else:
        X_2 = X_2.T
        for i in range(m):
            X_2[i, X[i,]] = 1

        return X_2, Y_2, id_to_word

if __name__ == '__main__':
    X_2, Y_2, id_to_word = get_input_output()