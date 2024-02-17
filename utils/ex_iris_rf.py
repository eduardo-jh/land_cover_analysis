#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import export_text, plot_tree


fn_save_trees = "/data/ssd/eduardojh/results/iris_tree.txt"

iris = load_iris()

print(iris.target_names)
print(iris.feature_names)

df = pd.DataFrame(iris.data, columns=iris.feature_names)
print(df.head())

df['target'] = iris.target
print(iris.target)

X_train, X_test, y_train, y_test = train_test_split(df.drop(['target'], axis='columns'), iris.target, test_size=0.2)

clf = RandomForestClassifier(n_estimators=50)
clf.fit(X_train, y_train)

for i, tree in enumerate(clf.estimators_):
    # A text representation
    tree_str = export_text(tree, feature_names=iris.feature_names, max_depth=10)
    with open(fn_save_trees[:-4] + '_' +  str(i).zfill(3) + '.txt', 'w') as f_tree:
        f_tree.write(tree_str)
    # A figure
    plot_tree(tree)
    plt.savefig(fn_save_trees[:-4] + '_' +  str(i).zfill(3) + '.png', bbox_inches='tight', dpi=600)