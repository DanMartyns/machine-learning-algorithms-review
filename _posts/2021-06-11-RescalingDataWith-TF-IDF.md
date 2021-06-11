---
layout: post
identifier: 20
title: "Rescaling the Data with tf–idf"
categories:
  - Working with Text Data
tags:
---

Instead of dropping features that are deemed unimportant, another approach is to rescale features by how informative we expect them to be. One of the most common ways to do this is using the *term frequency-inverse document frequency (tf-idf)* method. The intuition of this method is to give high weight to any term that appears often in a particular document, but not in very many documents, it is likely to be very descriptive of the content of that document.

*scikit-learn* implements the tf-idf method in two classes: **TFidfTransformer**, which takes in the sparse matrix output produced by **CountVectorizer** and transforms it, and **TfidfVectorizer**, which takes in the text data and does both the bag-of-words feature extraction and the tf-idf transformation. There are several variants of the tf-idf rescaling scheme. The tf-idf score for word *w* in the document *d* as implemented in both the **TfidfTransformer** and **TfidfVectorizer** classes is given by:

$$ tfidf(w,d) = tf \cdot log(\frac{N+1}{N_w + 1}) + 1$$

where N is the number of documents in the training set, $$N_w$$ is the number of documents in the training set that the word w appears in, and *tf (term frequency)* if the number of times that the word w appears in the query document d (the document you want to transform or encode). Both classes also apply L2 normalization after computing the tf-idf representation; in other words, they rescale the representation of each document to have Euclidean norm 1. Rescaling in this way means that the lenght of a document (the number of words) does not change the vectorized representation.

Because tf-idf actually makes use of the statistical properties of the training data, we will use a pipeline, to ensure the results of our grid search are valid. This leads to the following code:

```python
In:
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import make_pipeline
pipe = make_pipeline(TfidfVectorizer(min_df=5, norm=None),
 LogisticRegression())
param_grid = {'logisticregression__C': [0.001, 0.01, 0.1, 1, 10]}
grid = GridSearchCV(pipe, param_grid, cv=5)
grid.fit(text_train, y_train)
print("Best cross-validation score: {:.2f}".format(grid.best_score_))
```

```text
Out:
Best cross-validation score: 0.89
```

As you can see, there is some improvement when using tf-idf instead of just word counts. We can also inspect which words tf-idf found most important. Keep in mind that the tf–idf scaling is meant to find words that distinguish documents, but it is a purely unsupervised technique. So, “important” here does not necessarily relate to the “positive review” and “negative review” labels we are interested in.

```python
In:
vectorizer = grid.best_estimator_.named_steps["tfidfvectorizer"]
# transform the training dataset
X_train = vectorizer.transform(text_train)
# find maximum value for each of the features over the dataset
max_value = X_train.max(axis=0).toarray().ravel()
sorted_by_tfidf = max_value.argsort()
# get feature names
feature_names = np.array(vectorizer.get_feature_names())
print("Features with lowest tfidf:\n{}".format(
 feature_names[sorted_by_tfidf[:20]]))
print("Features with highest tfidf: \n{}".format(
 feature_names[sorted_by_tfidf[-20:]]))
```

```text
Out:
Features with lowest tfidf:
['poignant' 'disagree' 'instantly' 'importantly' 'lacked' 'occurred'
 'currently' 'altogether' 'nearby' 'undoubtedly' 'directs' 'fond' 'stinker'
 'avoided' 'emphasis' 'commented' 'disappoint' 'realizing' 'downhill'
 'inane']
Features with highest tfidf:
['coop' 'homer' 'dillinger' 'hackenstein' 'gadget' 'taker' 'macarthur'
 'vargas' 'jesse' 'basket' 'dominick' 'the' 'victor' 'bridget' 'victoria'
 'khouri' 'zizek' 'rob' 'timon' 'titanic']
```

Features with low tf-idf are those that either are very commonly used across documents or are only used sparingly, and only in very long documents. Interestingly, many of the high-tf-idf features actually identify certain shows or movies. These terms only appear in reviews for this particular show or franchise, but tend to appear very often in these particular reviews. This is very clear, for example, for "pokemon", "smallvile", and "doodlebops", but "scanners" here actually also referes to a movie title. These words are unlikely to help us in our sentiment classification task (unless maybe some franchises are universally reviewd positively or negatively) but certainly contain a lot of specific information about the reviews.

We can also find the words that have low inverse document frequency - that is, those that appear frequently and are therefore deemed less important. The inverse document frequency values found on the training set are stored in the *idf_* attribute.

```python
In:
sorted_by_idf = np.argsort(vectorizer.idf_)
print("Features with lowest idf:\n{}".form
```

```text
Out:
Features with lowest idf:
['the' 'and' 'of' 'to' 'this' 'is' 'it' 'in' 'that' 'but' 'for' 'with'
 'was' 'as' 'on' 'movie' 'not' 'have' 'one' 'be' 'film' 'are' 'you' 'all'
 'at' 'an' 'by' 'so' 'from' 'like' 'who' 'they' 'there' 'if' 'his' 'out'
 'just' 'about' 'he' 'or' 'has' 'what' 'some' 'good' 'can' 'more' 'when'
 'time' 'up' 'very' 'even' 'only' 'no' 'would' 'my' 'see' 'really' 'story'
 'which' 'well' 'had' 'me' 'than' 'much' 'their' 'get' 'were' 'other'
 'been' 'do' 'most' 'don' 'her' 'also' 'into' 'first' 'made' 'how' 'great'
 'because' 'will' 'people' 'make' 'way' 'could' 'we' 'bad' 'after' 'any'
 'too' 'then' 'them' 'she' 'watch' 'think' 'acting' 'movies' 'seen' 'its'
 'him']
```

As expected, these are mostly English Stop Words like "the" and "no". But some are clearly domain-specific to the movie reviews, like "movie", "film", "time", "story", and so on. Interestingly, "good", "great", and "bad" are also among the most frequent and therefore "least relevant" words according to the tf-idf measure, even though we might expect these to be very important for our sentiment analysis task.