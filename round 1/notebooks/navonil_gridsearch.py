optimisation = GridSearchCV()
parameters = [{'solver':  [‘newton-cg’, ‘lbfgs’, ‘liblinear’, ‘sag’, ‘saga’], 'C': [1,2,3,300,500], 'penalty': [‘l1’, ‘l2’, ‘elasticnet’, ‘none’]}]
search_model = optimisation(LogisticRegression(), parameters, scoring = 'accuracy')
search_model.fit(training_set_X_trans, training_set_Y)