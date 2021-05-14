---
layout: post
identifier: 11
title: "Gradient boosted regression trees (gradient boosting machines)"
categories:
  - Supervised Learning
subcategory:
  - Decision Trees
tags:
  - Supervised Learning
  - Regression
  - Classification
  - Strenghts
  - Weaknesses
---

The gradient boosted regression is also an essemble method that combines several decision trees to create a more powerful model. Despite the "regression" in the name, the gradient boosted regression can be use for regression or classification. In contrast with the random forest approach, the gradient boosted regression works by building tress in a serial way, where each tree tries to correct the predictions of the previous tree.

By default, there is no randomization in gradient boosted regression trees. Instead, strong pre-pruning is used. Gradient boosted trees often use very small trees, of deepth one to five, which makes the model smaller in terms of memory and makes predictions faster.

The **main ideia** behind gradient boosted regression **is to combine many simple models**, known as ***weak learners***. Each tree can only provide good predictions on part of the data, the predictions on the rest of data is iteratively improved by adding more trees.

Gradient boosted trees are frequently the winning entries in the machine learning competitions, and are widely use for industrial applications. They are more sensitive to parameter settings than random forests, but can provide better accuracy if the parameters are set correctly.

Besides the pre-pruning and the number of trees in the essemble, another important parameter is the *learning_rate*, which controls how strongly each tree tries to correct the mistakes of the previous trees. A higher learning rate means each tree can make a stronger corrections, allowing for more complex models. 

```python   
In:
from sklearn.ensemble import GradientBoostingClassifier

X_train, X_test, y_train, y_test = train_test_split(
 cancer.data, cancer.target, random_state=0)
 
gbrt = GradientBoostingClassifier(random_state=0)
gbrt.fit(X_train, y_train)

print("Accuracy on training set:{:.3f}".format(gbrt.score(X_train, y_train)))
print("Accuracy on test set:{:.3f}".format(gbrt.score(X_test, y_test)))
```
```text   
Out: 
Accuracy on training set: 1.000
Accuracy on test set: 0.958
```

As the training set accuracy is 100%, we are likely to be overfitting. To reduce overfitting, we could either apply stronger pre-pruning by limiting the maximum depth or lower the learning rate:

```python   
In:
gbrt = GradientBoostingClassifier(random_state=0, max_depth=1)
gbrt.fit(X_train, y_train)

print("Accuracy on training set:{:.3f}".format(gbrt.score(X_train, y_train)))
print("Accuracy on test set:{:.3f}".format(gbrt.score(X_test, y_test)))
```
```text   
Out: 
Accuracy on training set: 0.991
Accuracy on test set: 0.972
```
```python   
In:
gbrt = GradientBoostingClassifier(random_state=0, learning_rate=0.01)
gbrt.fit(X_train, y_train)

print('Accuracy on training set:{:.3f}'.format(gbrt.score(X_train, y_train)))
print('Accuracy on test set:{:.3f}'.format(gbrt.score(X_test, y_test)))
```
```text   
Out: 
Accuracy on training set: 0.988
Accuracy on test set: 0.965
```

Both methods of decreasing the model complexity reduced the training set accuracy, as expected. In this case, lowering the maximum depth of the trees provided a significant improvement of the model, while lowering the learning rate only increased the generalization performance slightly.

As both gradient boosting and random forests perform well on similar kinds of data, a common approach is to first try random forests, which work quite robustly. If random forests work well but prediction time is at a premium, moving to gradient boosting often helps.

## Strenghts, Weaknesses, and Parameters

Gradient boosted decision trees are among the most powerful and widely used models for supervised learning. Their main drawback is that they require careful tuning of the parameters and may take a long time to train. Similarly to other tree-based models, the algorithm works well without scaling and on a mixture of binary and continuous features. As with other tree-based models, it also often does not work well on high-dimensional sparse data.

The main parameters of gradient boosted tree models are the number of trees, *n_estimators*, and the *learning_rate*, which controls the degree to which each tree is allowed to correct the mistakes of the previous trees. These two parameters are highly interconnected, **as a lower *learning_rate*** means that more trees are needed to build a model of similar complexity. In contrast to random forests, where a higher *n_estimators* value is always better, increasing *n_estimators* in gradient boosting leads to a more complex model, which may lead to overfitting. A common practice is to fit *n_estimators* depending on the time and memory budget, and then search over different *learning_rates*.

Another important parameter is *max_depth* (or alternatively *max_leaf_nodes*), to reduce the complexity of each tree. Usually *max_depth* is set very low for gradient boosted models, often not deeper than five splits.