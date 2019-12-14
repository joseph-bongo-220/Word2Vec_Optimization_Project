import pickle
import numpy as np
import pandas as pd
import random
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from prettytable import PrettyTable

def evaluate_embeddings():
    with open("gradient_descent.pkl", 'rb') as f:
        gd = pickle.load(f) 
    with open("gradient_descent_fixed.pkl", 'rb') as f:
        gd_fixed = pickle.load(f) 
    with open("gradient_descent_backtracking_line_search.pkl", 'rb') as f:
        gd_ls = pickle.load(f) 
    with open("adam.pkl", 'rb') as f:
        adam = pickle.load(f) 

    emb = [gd, gd_fixed, gd_ls, adam]

    test_words = ["bath", "weather", "offensive", "ambassador", "divorced", "innocent", "suburban", "grand"]

    t = PrettyTable(['Method', 'Word', '1', '2', '3', '4', '5'])

    for word in test_words:
        similarity = dict()
        for w, vec in gd.items():
            similarity.update({w: np.dot(gd[word], vec)/ (np.linalg.norm(gd[word])*np.linalg.norm(vec))})
        sim = list(similarity.items())
        sim = sorted(sim, key = lambda x: x[1], reverse = True)[1:6]
        sim = [(k, round(v, 3)) for k,v in sim]
        t.add_row(["Gradient Desent", word, sim[0], sim[1], 
        sim[2], sim[3], sim[4]])

    for word in test_words:
        similarity = dict()
        for w, vec in gd_fixed.items():
            similarity.update({w: np.dot(gd_fixed[word], vec)/ (np.linalg.norm(gd_fixed[word])*np.linalg.norm(vec))})
        sim = list(similarity.items())
        sim = sorted(sim, key = lambda x: x[1], reverse = True)[1:6]
        sim = [(k, round(v, 3)) for k,v in sim]
        t.add_row(["Gradient Desent - Fixed", word, sim[0], sim[1], 
        sim[2], sim[3], sim[4]])    

    for word in test_words:
        similarity = dict()
        for w, vec in gd_ls.items():
            similarity.update({w: np.dot(gd_ls[word], vec)/ (np.linalg.norm(gd_ls[word])*np.linalg.norm(vec))})
        sim = list(similarity.items())
        sim = sorted(sim, key = lambda x: x[1], reverse = True)[1:6]
        sim = [(k, round(v, 3)) for k,v in sim]
        t.add_row(["Gradient Desent - Line Search", word, sim[0], sim[1], 
        sim[2], sim[3], sim[4]])    

    for word in test_words:
        similarity = dict()
        for w, vec in adam.items():
            similarity.update({w: np.dot(adam[word], vec)/ (np.linalg.norm(adam[word])*np.linalg.norm(vec))})
        sim = list(similarity.items())
        sim = sorted(sim, key = lambda x: x[1], reverse = True)[1:6]
        sim = [(k, round(v, 3)) for k,v in sim]
        t.add_row(["Adam", word, sim[0], sim[1], 
        sim[2], sim[3], sim[4]])  

    print(t)

def get_knn_accuracy(input_gd, pos):
    input_gd = pd.DataFrame.from_records(input_gd)

    X_train, X_test, y_train, y_test = train_test_split(input_gd, pos, test_size=0.40, random_state = 122019)

    scaler = StandardScaler()
    scaler.fit(X_train)
    
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    knn = KNeighborsClassifier(n_neighbors=6)
    knn.fit(X_train, y_train)

    y_pred = knn.predict(X_test)

    r = (y_pred == y_test)
    return len(r[r==True])/len(r)


def classification_task():
    with open("gradient_descent.pkl", 'rb') as f:
        gd = pickle.load(f) 
    with open("gradient_descent_fixed.pkl", 'rb') as f:
        gd_fixed = pickle.load(f) 
    with open("gradient_descent_backtracking_line_search.pkl", 'rb') as f:
        gd_ls = pickle.load(f) 
    with open("adam.pkl", 'rb') as f:
        adam = pickle.load(f) 

    words = ["grand", "assault", "smile", "praise", "thanks", "win", "loser", "winner", "fight", "arrest",
    "happy", "mad", "upset", "unhappiest", "evil", "poor", "love", "lover", "like", "lovely",
    "bad", "good", "great", "honored", "dead", "impressive", "fail", "fear", "broken", "criminal",
    "unfair", "gross", "fun", "fair", "kind", "nice", "fine", "guilty", "hate", "ill",
    "health", "paradise", "perfect", "just", "wrong", "sunny", "rain", "special", "yes", "no",
    "old", "young", "kill", "murder", "wonderful", "game", "offensive", "quit", "accept", "deny"]
    pos = [1, 0, 1, 1, 1, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 0, 0, 0, 0,
     0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 0, 1, 0, 1, 1, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0]

    print(str(sum(pos)))

    input_gd = [gd[x] for x in words]
    input_gd_fixed = [gd_fixed [x] for x in words]
    input_gd_ls = [gd_ls[x] for x in words]
    input_adam = [adam[x] for x in words]
    
    emb = [input_gd, input_gd_fixed, input_gd_ls, input_adam]

    t = PrettyTable(['Method', 'Accuracy'])
    t.add_row(["Gradient Descent", get_knn_accuracy(input_gd, pos)])
    t.add_row(["Gradient Descent - Fixed", get_knn_accuracy(input_gd_fixed, pos)])
    t.add_row(["Gradient Descent - Line Search", get_knn_accuracy(input_gd_ls, pos)])
    t.add_row(["Adam", get_knn_accuracy(input_adam, pos)])
    print(t)

if __name__ == "__main__":
    evaluate_embeddings()
    classification_task()