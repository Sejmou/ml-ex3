from sklearn import metrics
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


from sklearn import neighbors
from sklearn import naive_bayes
from sklearn import tree
from sklearn import ensemble
from sklearn import neural_network

import time
from datetime import timedelta

def evaluate_feature(X_train, y_train, X_test, y_test):
    ping = time.time()
    # set up a number of classifiers
    classifiers = [
                neighbors.KNeighborsClassifier(),
                neural_network.MLPClassifier(),
                naive_bayes.GaussianNB(), 
                tree.DecisionTreeClassifier(),
                ensemble.RandomForestClassifier(random_state=123)
                ]
    params = [
            { 
                'clf__n_neighbors': [i*5 for i in range(1, 6)],
                'clf__weights':  ('uniform', 'distance')
            },            
            {
                'clf__hidden_layer_sizes': [(100,),(50,)],
                'clf__learning_rate': ["adaptive"],
                'clf__learning_rate_init': [0.01],
                'clf__activation': ["logistic", "relu", "tanh"]
            },
            {},
            {
                'clf__criterion': ('gini', 'entropy'),
                'clf__splitter': ('best', 'random')
            },
            {
                'clf__n_estimators': [i*50 for i in range(1, 3)],
                'clf__criterion': ('gini', 'entropy'),
            }
    ]

    # Now iterate over classifiers, train and evaluate:
    best_cl = None
    best_acc = 0
    best_predict_time = None
    for classifier, param in zip(classifiers, params):
        print(f"Trying {classifier}")
        pipe = Pipeline([('scaler', StandardScaler()),('clf', classifier)])
        clf = GridSearchCV(pipe, param, cv=5, n_jobs=4)
        clf.fit(X_train, y_train)
        print(f"Fitted, best are {clf.best_params_} with cross val score of {clf.best_score_}.")
        ping_pred = time.time()
        pred = clf.predict(X_test)
        pong_pred = time.time()
        acc = metrics.accuracy_score(y_test, pred)
        print(f"Accuracy on Test Set is {acc}")
        if acc > best_acc:
            best_acc = acc
            best_cl = (classifier, clf.best_params_)
            best_predict_time = pong_pred - ping_pred
    pong = time.time() - ping
    hours, rem = divmod(pong, 3600)
    minutes, seconds = divmod(rem, 60)
    message = "{:0>2}:{:0>2}:{:05.2f}".format(int(hours), int(minutes), seconds)
    print(f"Best Classifier is {best_cl} with an accuracy of {best_acc}, predicting took {best_predict_time} seconds and this whole process took {message}")