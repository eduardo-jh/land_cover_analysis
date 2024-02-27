#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import export_text, plot_tree
from joblib import dump, load


fn_save_trees = "/data/ssd/eduardojh/results/iris_tree.txt"
fn_trained_rf = "/data/ssd/eduardojh/results/trained_model.joblib"

iris = load_iris()

print(iris.target_names)
print(iris.feature_names)

df = pd.DataFrame(iris.data, columns=iris.feature_names)
print(df.head())

df['target'] = iris.target
print(iris.target)

X_train, X_test, y_train, y_test = train_test_split(df.drop(['target'], axis='columns'), iris.target, test_size=0.2)

n_trees = 50
print(f"Training random forest with {n_trees} estimators...")

clf = RandomForestClassifier(n_estimators=n_trees)
clf.fit(X_train, y_train)

print("Saving trained model...", end='')
dump(clf, fn_trained_rf)
print("done!")

print("Saving tree: ", end='')
for i, tree in enumerate(clf.estimators_):
    # A text representation
    print(f"{i+1}/{n_trees}", end='')
    print("  text representation...", end='')
    tree_str = export_text(tree, feature_names=iris.feature_names, max_depth=len(iris.feature_names), decimals=0)
    with open(fn_save_trees[:-4] + '_' +  str(i).zfill(3) + '.txt', 'w') as f_tree:
        f_tree.write(tree_str)
    # A figure
    print("  graphical representation...")
    plt.figure()
    plot_tree(tree)
    plt.savefig(fn_save_trees[:-4] + '_' +  str(i).zfill(3) + '.eps', format='eps', bbox_inches='tight')

print("Loading trained model...")
trained_clf = load(fn_trained_rf)
print("Predicting...")
predictions = trained_clf.predict(X_test)

print("All done! ;-)")