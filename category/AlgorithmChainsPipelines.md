---
layout: category
title: Algorithm Chains and Pipelines
---

For many machine learning algorithms, the particular representation of the data that you provide is very important. This starts with scaling the data and combining features by hand and goes all the way to learning features using unsupervised machine learning. Consequently, most machine learning applications require not only the application of a single algorithm, but the chaining together of many different processing steps and machine learning models. In this section, we will cover how to use the **Pipeline** class to simplify the process of building chains of transformations and models. In particular, we will see how we can combine **Pipeline** and **GridSearchCV** to search over parameters for all processing steps at once.