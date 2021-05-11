---
layout: post
identifier: 10
title: "Random Forests"
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
last_modified_at: {{ page.last_modified_at }}
---

A main drawback of decision trees is that they tend to overfit the training data. Random forests are one way to address this problem. A random forest is essentially a collection of decision trees, where each tree is slightly different from the others. The idea behind this is that each tree might do a relatively good job of predicting, but will likely overfit in different ways, we can reduce the amount of overfitting by averaging their results.

## How to build a Random Forest?

First of all, you need to **decide the number of trees to build** (the *n_estimators* parameter). These trees will be built completely independently from each other, and the algorithm will make different random choices for each tree to make sure the trees are distinct. To build a tree, we first take what we called a *bootstrap sample* of our data. That is, from out *n_samples* data points, we repeatedly draw an example randomly with replacement, *n_sample* times. This will create a dataset that is a big as the original dataset, but some data points will be missing from it, and some will be repeated.
