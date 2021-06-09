---
layout: post
identifier: 15
title: "Summary and Outlook"
categories:
  - Supervised Learning
tags:
  - Supervised Learning
  - Generalization
  - Underfitting
  - Overfitting
  - Regression
  - Classification
---

We started this chapter with a discussion of model complexity, then discussed *generalization*, or learning a model that is able to perform well on new, previously unseen data. This led us to the concepts of *underfitting*, which describes a model that cannot capture the variations present in the training data, and *overfitting*, which describes a model that focuses too much on the training data and is not able to generalize to new data very well.

We then discussed a wide array of machine learning models for classification and regression, what their advantages and disadvantages are, and how to control model complexity for each of them. We saw that for many of the algorithms, setting the right parameters is important for good performance. Some of the algorithms are also sensitive to how we represent the input data, and in particular to how the features are scaled. Therefore, blindly applying an algorithm to a dataset without understanding the assumptions the model makes and the meanings of the parameter settings will rarely lead to an accurate model.

This chapter contains a lot of information about the algorithms, and it is not necessary for you to remember all of these details for the following chapters. However, some knowledge of the models described here - and which to use in a specific situation - is important for successfully applying machine learning in practice. Here is a quick summary of when to use each model:

*Nearest Neighbors*
 - For small datasets, good as a baseline, easy to explain

*Linear Models*
 - Go-to as a first algorithm to try, good for very large datasets, good for very high-dimensional data.

*Naive Bayes*
 - Only for classification. Even faster than linear models, good for very large datasets and high-dimensional data. Often less accurate than linear models.

*Decision trees*
 - Very fast, don't need scalling of the data, can be visualized and easy explained.

*Random Forests*
 - Nearly always perform betten than single decision tree, very robust and powerful. Don't need scaling of data. Not good for very high-dimensional sparse data.

*Gradient Boosted decision trees*
 - Often slightly more accurate than random forests. Slower train but faster to predict than random forests, and smaller in memory. Need more parameter tunning than random forests.

*Support Vector Machines*
 - Powerful for medium-sized datasets of features with similar meaning. Require scaling of data, sensitive to parameters.

*Neural Networks*
 - Can build bery complex models, particularly for large datasets. Sensitive to scaling of the data and to the choice of parameters. Large models need a long time to train.

When working with a new dataset, **it is in general a good idea to start with a simple model, such as a linear model or a naive Bayes or nearest neighbors classifier**, and see how far you can get. **After understanding more about the data, you can consider moving to an algorithm that can build more complex models, such as random forests, gradient boosted decision trees, SVMs, or neural networks.**