# language-id
Identifying languages using an n-gram model and logistic regression

## Purpose
This simple snippet of code allows you to train models for langauge identification using logistic regression and the Wili-2018 dataset (paper here: https://arxiv.org/abs/1801.07779). This dataset is composed of sentences from Wikipedia articles in various languages.  Essentially, after you select which languages you want to distinguish between, this program fetches the relevant sentences from the training and test datasets. Then, using those as a corpus, it finds all of the possible n-grams (you can choose the n) in those languages. It then represents each sentence in the corpus as a rather large vector containing count of each n-gram in that sentence. This is used as training data for a logistic regression model.

## Usage
This entire project can be run from the combined.py file. Using the command line options you can specify a variety of things.

|Flag|Description|
|---|---|
|-p|The path to the train and test files from the Wili-2018 dataset|
|-n|Enter values in the format "x,y" (no spaces or quotes). The n-grams used will be n in range [x,y]|
|-d|The minimum count that each n-gram must have in order to be counted (if it falls below this value, those n-grams will be ignored)|
|-s|The file in which to save the model. If this is left unspecified, the model is not saved.|


This model requires: sklearn, pickle
