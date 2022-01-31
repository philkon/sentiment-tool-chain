# This repository is no longer maintained!

Please see: 

URL: https://www.melusinapress.lu/read/ezpg-wk34

GIT: https://gitlab.uni.lu/melusina/vdhd/koncar_sentiment

# A Sentiment Tool Chain for Languages of the 18<sup>th</sup> Century

This repository provides a ready-to-use and adaptable tool chain to create sentiment dictionaries for and to conduct sentiment analysis with the languages of the 18<sup>th</sup> century.
At its current state, the sentiment analysis part supports French, Italian and Spanish, but it can easily be applied and transferred to other languages by using the dictionary creation part.

## Table of Contents
* [Introduction to Sentiment Analysis](#introduction-to-sentiment-analysis)
* [Tool Chain](#tool-chain)
* [Requirements](#requirements)
  * [Python](#python)
  * [Dataset](#dataset)
  * [Hardware](#hardware)
* [Creating Sentiment Dictionaries](#creating-sentiment-dictionaries)
  * [Overview](#overview)
  * [Seed Words](#seed-words)
    * [Extraction](#extraction)
    * [Annotation](#annotation)
    * [Selection](#selection)
  * [Word Embeddings](#word-embeddings)
    * [Word2Vec](#word2vec)
    * [Grid Search](#grid-search)
    * [Model Evaluation](#model-evaluation)
  * [Classification](#classification)
    * [KNN-Classifier](#knn-classifier)
    * [Classifier Evaluation](#classifier-evaluation)
* [Sentiment Analysis for the 18<sup>th</sup> Century](#sentiment-analysis-for-the-18th-century)
  * [Dictionaries](#dictionaries)
    * [Manually Annotated](#manually-annotated)
    * [Computationally Annotated](#computationally-annotated)
    * [Computationally Annotated and Corrected](#computationally-annotated-and-corrected)
  * [Computing Sentiment](#computing-sentiment)
  * [Data Preparation](#data-preparation)
  * [Example Texts](#example-texts)
  * [Analysis](#analysis)
    * [Examples](#examples)
  * [Where to Go From Here](#where-to-go-from-here)
* [Notes](#notes)
  * [Citation Information](#citation-information)
  * [The DiSpecs Project](#the-dispecs-project)
  * [Development and Repository](#development-and-repository)
  * [Word Annotations](#word-annotations)
  * [Acknowledgments](#acknowledgments)
* [References](#references)

## Introduction to Sentiment Analysis
Sentiment analysis is a common task in natural language processing (NLP) and aims for the automatic and computational identification of emotions, attitudes and opinions expressed in textual data (Pang et al. 2008).
Most of basic approaches rely on *sentiment dictionaries* comprising lists of words for which the sentiment polarity (either positive or negative) is known.
These dictionaries are then used to assess the sentiment of text, for example, by considering occurrences of words in the text that are also in the dictionary.
The assignment of sentiment could either be categorical (e.g., *positive*, *negative*, *neutral*) or continuous through a score of a given range (e.g., ranging form -1 to 1, where values close to -1 represent a negative sentiment, values close to +1 represent a positive sentiment and values around 0 represent a neutral sentiment).
Note that sentiment can be computed on different levels (e.g., document, paragraph, sentence).

We now demonstrate sentiment analysis based on dictionaries with an example.
Let us assume we want to compute the sentiment of the following three sentences:

* Today was a good day.
* I hate getting up early.
* This is both funny and sad at the same time.

Further, we have the following (exemplary) sentiment dictionary:

Word | Sentiment 
--- | ---
good | positive
bad | negative
hate | negative
funny | positive
sad | negative
happy | positive

As you can see, the dictionary maps every word in it to a sentiment.
In this case, the sentiment is either positive or negative, but as mentioned above, it could also include other categories or be represented by a numerical value.

We now check if any of the words from the dictionary occur in either of our sentences and compute their sentiment by a simple majority vote:
* If there are more positive words than negative words in a sentence, the sentence is positive.
* If there are more negative words than positive words in a sentence, the sentence is negative.
* If there are no dictionary words or an equal number of positive and negative words in a sentence, the sentence is neutral.

Note that this majority vote is just one straightforward way for considering word occurrences.
Typically, more complex algorithms are used to derive sentiment from word occurrences. 

Considering this for the three sentences of our example, we observe:

* Today was a **good** day. -> positive sentiment (because the sentence contains one positive word from the dictionary)
* I **hate** getting up early. -> negative sentiment (because the sentence contains one negative word from the dictionary)
* This is both **funny** and **sad** at the same time. -> neutral sentiment (because the sentence contains one positive and one negative word from the dictionary)

Usually, such dictionaries are manually annotated and curated which makes their creation very complex and time-consuming.
Examples for dictionary based approaches include SentiStrength (Thelwall et al. 2010) or VADER (Hutto and Gilbert 2014), both specifically introduced for short, English texts originating from social media platforms, such as Twitter or Facebook.
Further, machine learning approaches for sentiment classification have been studied (Maas et al. 2011; Pang et al. 2002; Ye et al. 2009), including more sophisticated neural networks (Zhang et al. 2018).
However, these advanced methods are often harder to interpret than dictionary based approaches.

As most of the studies in the field of sentiment analysis (or, in general, all NLP areas) focus on Modern English, we encounter a significant scarcity of dictionaries suitable for non-English languages as well as for languages of earlier times.
This is partly due to linguistic barriers, such as different word meaning or spelling in those times, which prevent quick transfers of existing models.

Our proposed sentiment tool chain aims to fill the gap of missing approaches for languages of the 18<sup>th</sup> century and can serve as a stepping stone towards more sophisticated sentiment analysis models that enable us to better understand and interpret historic texts.

### Tool Chain
The proposed tool chain comprises two different parts: **(i) the creation of sentiment dictionaries** and **(ii) the actual sentiment analysis**.
The first part is not mandatory and provides an approach to create sentiment dictionaries that are applicable to the languages of the 18<sup>th</sup> century.
The second part provides ready-to-use methods to analyze sentiment of texts written in the French, Italian and Spanish of the 18<sup>th</sup> century.
If you want to use the analyze sentiment part in other languages, you will have to create your own dictionaries as described in the [Creating Sentiment Dictionaries](#creating-sentiment-dictionaries) section. Note that this step requires a *manual* annotation.

### Requirements

We recommend you to install [Anaconda](https://www.anaconda.com/), as it comes pre-bundled with most required packages.

#### Python
If you simply want to use our dictionaries and Jupyter Notebooks to analyze sentiment of your texts, you need to have the following additional Python packages installed:
* pandas 1.0.1
* Matplotlib 3.2.0
* Seaborn 0.11.0
* tqdm 4.50.2
* nltk 3.5
* Jupyter Notebook 6.1.4 or Jupyter Lab 2.2.6
* ipywidgets 7.5.1

In order to create dictionaries yourself, you need to have the following additional Python packages installed:
* pandas 1.0.1
* gensim 3.8.3 (needs to be [installed separately](https://anaconda.org/anaconda/gensim); you best use `pip install gensim` in an Anaconda prompt to install gensim as the conda package is not up-to-date and cannot be installed with Anaconda running Python 3.8)
* sklearn 0.23.2
* nltk 3.5
* spacy 2.2.3 (needs to be [installed separately](https://anaconda.org/conda-forge/spacy))
* stop-words 2018.7.23 (needs to be [installed separately](https://anaconda.org/conda-forge/stop-words))
* Jupyter Notebook 6.1.4 or Jupyter Lab 2.2.6
* ipywidgets 7.5.1

Note that we tested our notebooks with the versions stated above.
While older and newer versions may work, the outcome may be impaired.
 
#### Dataset
If you want to create dictionaries based on your own data, make sure that you have a decent amount of text.
The more text, the better the output.
Also, make sure that you cleaned your data and that each document is contained in a single *.txt* file with UTF-8 encoding.

Our ready-to-use dictionaries and models base on Spectator periodicals published during the 18<sup>th</sup> century.
In particular, we leverage **The Spectators in the international context**, a digital scholarly edition project which aims on building a central repository for spectator periodicals (Ertler et al. 2011, Scholger 2018).
The annotated periodicals follow the XML-based Text Encoding Initiative (TEI) standard (Consortium 2020), which provides a vocabulary on how to represent texts in digital form, and are publicly available through the [digital edition](https://gams.uni-graz.at/spectators).
This dataset contains multiple languages, but we set our focus on French, Italian and Spanish, as these three languages have the largest collections.
For this purpose, we extracted texts from TEI encoded files into plain *.txt* files.

#### Hardware
Please keep in mind that your machine needs adequate hardware depending on the amount of text you want to consider.
This is especially important for the dictionary creation tool chain (e.g., we used a machine with 24 cores and 750 GB RAM and computations still took up to three days).
If you just want to analyze sentiment using existing dictionaries, a computer with common hardware should suffice.

## Creating Sentiment Dictionaries
This section covers a series of Jupyter Notebooks which can be easily used by everyone to create sentiment dictionaries for their own projects.
We provide a detailed step-by-step manual of our proposed tool chain as well as examples for the methods used in it.
Please refer to the [Sentiment Analysis for the 18<sup>th</sup> Century](#sentiment-analysis-for-the-18th-century) section if you want to use our ready-to-use dictionaries for the French, Italian and Spanish of the 18<sup>th</sup> century.

### Overview
<p align="center">
  <img src="images/dictionary_creation_overview.png?raw=true" alt="Dictionary Creation Overview"/>
</p>

Our tool chain for creating sentiment dictionaries comprises three major steps, each of them described in detail in the following sections.
Overall, we first need to generate a set of seed words which serve as a basis to automatically expand sentiment to other words in the text corpus.
For this expansion, we train word embeddings to capture the context of individual words and use them in a classification task to transfer sentiment of seed words to other words in similar contexts.
Note that this process is suited for a plethora of languages and not limited to the languages of the 18<sup>th</sup> century.

### Seed Words
Our sentiment dictionary creation depends on seed words for which the sentiment is known.
Based on these seed words, we can automatically transfer sentiment to other words, allowing us to circumvent a more tedious and time-consuming annotation process.
This step comprises two parts: First, we need to extract seed words and second, we need to manually annotate them.

#### Extraction
* You can find the Jupyter Notebook for the extraction of seed words in [dictionary_creation/seed_words/01_seed_words_extraction.ipynb](dictionary_creation/seed_words/01_seed_words_extraction.ipynb).

In this step, we extract the 3000 most frequent words from the entire text corpus (with a maximum document frequency of 80%).
Focusing on most frequent words allows us to achieve a good coverage as seed words that do not occur frequently in our texts are of no use to us.

#### Annotation
* You can find the Jupyter Notebook for the annotation of seed words in [dictionary_creation/seed_words/02_seed_words_annotation.ipynb](dictionary_creation/seed_words/02_seed_words_annotation.ipynb).
* You can find our annotated seed words for French, Italian and Spanish in [dictionary_creation/seed_words/ready_to_use/](dictionary_creation/seed_words/ready_to_use/).

This step requires the manual annotation of previously extracted seed words.
The Jupyter Notebook provides an easy way to generate *.csv* files that can be opened and annotated in a spreadsheet program of your choice (e.g., Excel or LibreOffice Calc).
Please annotate seed words in the *sentiment* column and make sure you use either of the following three sentiment classes: positive, negative or neutral.
For example:

word | sentiment
--- | ---
good | positive
bad	| negative
house |	neutral

Once you (or your annotators) are finished, make sure to save the file using the *.csv* file format.

Note that you must use one distinct annotation file for each annotator.
During the creation of *.csv* files in the Jupyter Notebook, you are asked for the name of the annotator which will be used for the filename.
This allows you to create separate annotation files (one for each of your annotators).

#### Selection
* You can find the Jupyter Notebook for the selection of seed words in [dictionary_creation/seed_words/03_seed_words_selection.ipynb](dictionary_creation/seed_words/03_seed_words_selection.ipynb).

You should use at least three different annotators as sentiment is very subjective.

For our ready-to-use dictionaries, we decided to use three independent annotators that are experts in humanities and have substantial knowledge of the text corpora.
We combine their annotations by using a majority vote: We only keep words for which at least two annotators have equal annotations and remove remaining words.
Our Jupyter Notebook provides a ready-to-use implementation to select seed words based on a majority vote.

### Word Embeddings
* You can read more about word embeddings in the corresponding [Wikipedia article](https://en.wikipedia.org/wiki/Word_embedding).

While computers are very good in math, they have hard times in understanding text or words.
To counteract this problem, one needs to transform text or words into numerical forms.
One way to achieve this are word embeddings, which represent words by vectors.
There are frequency based word embeddings, such as count vectors or TF-IDF vectors, as well as prediction based word embeddings, such as word2vec.

The following example demonstrates word embeddings based on count vectors.
Consider the following three sentences:

* Word embeddings are very cool.
* My students are cool.
* The teacher is also cool.

Let us create a table in which each row represents one of our three sentences and each column represents a distinct word occurring in either of our three sentences.
For each cell, we have 1 if the word occurs in the respective sentence or 0 otherwise:

| | word | embeddings | are | very | cool | my | students | the | teacher | is | also |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Sentence 1 | 1 | 1 | 1 | 1 | 1 | 0 | 0 | 0 | 0 | 0 | 0 |
| Sentence 2 | 0 | 0 | 1 | 0 | 1 | 1 | 1 | 0 | 0 | 0 | 0 |
| Sentence 3 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 1 | 1 | 1 | 1 |

We can then find the vector representation of a word by considering the values in its column.
For example, the vector of the word *are* is `[1, 1, 0]` and of the word *cool* is `[1, 1, 1]`.

Besides mapping words to numbers, word embeddings also capture the context of a word in a document, semantic similarity as well as relations with other words.
As such, vectors of words that are frequently used together in texts are also very close to each other in the vector space.
Contrary, vectors of words that are never or only minimally used in a similar context should be very distant to each other.
This suits perfectly for our dictionary creation process and based on the vector representation of words, we can automatically transfer sentiment form our seed words to other words.

We point the interested reader to the following YouTube videos for in-depth explanations of word embeddings:

<p align="center">
<a href="http://www.youtube.com/watch?feature=player_embedded&v=gQddtTdmG_8
" target="_blank"><img src="http://img.youtube.com/vi/gQddtTdmG_8/0.jpg" 
alt="Video 1" width="240" height="180" border="10" /></a>
</p>
<p align="center">
<a href="http://www.youtube.com/watch?feature=player_embedded&v=oUpuABKoElw
" target="_blank"><img src="http://img.youtube.com/vi/oUpuABKoElw/0.jpg" 
alt="Video 2" width="240" height="180" border="10" /></a>
</p>
<p align="center">
<a href="http://www.youtube.com/watch?feature=player_embedded&v=5PL0TmQhItY
" target="_blank"><img src="http://img.youtube.com/vi/5PL0TmQhItY/0.jpg" 
alt="Video 3" width="240" height="180" border="10" /></a>
</p>

#### Word2Vec
* You can read more about word2vec in the corresponding [Wikipedia article](https://en.wikipedia.org/wiki/Word2vec).

Word2Vec (Mikolov et al. 2013) is a state-of-the-art method to compute word embeddings.
It relies on a two-layer neural network to train word vectors that capture the linguistic contexts of words.
There are two different model architectures: continuous bag-of-words (CBOW) or skip-gram.
The former predicts a word from a given context whereas the latter predicts the context from a given word.
Both have their advantages and disadvantages regarding the size of the underlying text corpora, which is why we consider both of them in our tool chain.

The following excellent YouTube video of a Stanford lecture provides a detailed explanation of word2vec, the mathematics behind it and the two different architectures:
<p align="center">
<a href="http://www.youtube.com/watch?feature=player_embedded&v=ERibwqs9p38
" target="_blank"><img src="http://img.youtube.com/vi/ERibwqs9p38/0.jpg" 
alt="Video 2ord2vec" width="240" height="180" border="10" /></a>
</p>

Before training your word2vec model, you need to preprocess your texts.
Fortunately, the amount of required preprocessing for word2vec is very minimal.
In the Jupyter Notebook described in the next section, we simply remove stop words and extract individual sentences of texts.
The latter is important because individual sentences are the required input form for the word2vec implementation of gensim.

#### Grid Search
* You can find the Jupyter Notebook for training word2vec models in [dictionary_creation/word_embeddings/01_grid_search.ipynb](dictionary_creation/word_embeddings/01_grid_search.ipynb).
* You can read more about grid search in the corresponding [Wikipedia article](https://en.wikipedia.org/wiki/Hyperparameter_optimization#Grid_search).

Since word2vec has many different hyperparameters (that all have an impact on the resulting word embeddings), we need to tune them to achieve best possible performance.
One way to optimize hyperparameters is to conduct a *grid search*.
Here, we simply define a set of possible hyperparameters and train one individual model for each possible hyperparameter combination.
We then evaluate each model and pick the one that yielded the best performance. 
Our Jupyter Notebook contains a selection of possible hyperparameters, but further adjustments may be necessary based on your needs.
Please refer to the [documentation](https://radimrehurek.com/gensim/models/word2vec.html#gensim.models.word2vec.Word2Vec) of the word2vec implementation of gensim.

All models generated this way are stored for later use.
Depending on your text corpus and the number of hyperparameter combinations, the filesize of stored models can take multiple gigabytes.

#### Model Evaluation
* You can find the Jupyter Notebook for the evaluation of word2vec models in [dictionary_creation/word_embeddings/02_evaluation.ipynb](dictionary_creation/word_embeddings/02_evaluation.ipynb).
* You can find our evaluation word pairs for French, Italian and Spanish in the directory [dictionary_creation/word_embeddings/ready_to_use/word_pairs/](dictionary_creation/word_embeddings/ready_to_use/word_pairs/).
* You can find our evaluated and best performing word2vec models for French, Italian and Spanish in the directory [dictionary_creation/word_embeddings/ready_to_use/models/](dictionary_creation/word_embeddings/ready_to_use/models/).

In this step we evaluate our previously generated word2vec models.
We do this to find the hyperparameter combination that reflects relations between words the most accurate.
This can, for example, be accomplished through lists of manually annotated word pairs, in which every word pair was assigned with a relation score.
In our case, this score ranges from 0 to 10, where 0 represents no similarity and 10 represents absolute similarity.
For example:

* old & new -> 0
* easy & hard -> 1.23
* beautiful & wonderful -> 7.15
* rare & scarce -> 9.89

Usually, such lists are manually annotated and, thankfully, we can adapt previously existing lists for French, Italian and Spanish (Freitas et al. 2016; [GitHub Repository](https://github.com/siabar/Multilingual_Wordpairs)).
For that, we filter all words that are not existing in our texts, extend lists with spelling variations of the 18<sup>th</sup> century as well as check if relations scores are meaningful and also applicable to the languages of the 18<sup>th</sup> century.

Using these lists, we can compute Pearson and Spearman correlation coefficients (see [Wikipedia article](https://en.wikipedia.org/wiki/Correlation_coefficient)) between scores of word pairs and similarities of respective word vectors from the models.
We select the model for which the correlation coefficients are the highest and report the following values for respective languages:

Language | Pearson Rho 
--- | --- 
French | 0.402 
Italian | 0.157
Spanish | 0.310

Our resulting word embeddings for French, Italian and Spanish can be used for your own projects.
Just download the desired pickled word2vec model and load it in Python:

```python
import pickle
from gensim.models import Word2Vec

path_to_model = "" # set the path to where you saved the model (e.g., path_to_model="french.p")

with open(path_to_model, "rb") as handle:
    model = pickle.load(handle)
```

Note that due to file size limitations on GitHub, we moved pickled models to Google Drive.

**A word of caution:** Never unpickle **untrusted** data! While we guarantee that our pickled models are safe to use, you can never know what is in a pickled file. Please keep that in mind when dealing with pickled files :)

### Classification

In this step, we use the generated word embeddings to transfer the sentiment from our seed words to other words that appear in a similar context of seed words.
We use a *k*-nearest neighbors classifier that considers the distances between word vectors.
Remember that our word embeddings keep words that appear frequently in a similar context close together and words that are not related very distant to each other.
Thus, this approach is perfectly suited for our context based transfer of sentiment.

#### KNN Classifier
* You can read more about the *k*-nearest neighbor classifier in the corresponding [Wikipedia article](https://en.wikipedia.org/wiki/K-nearest_neighbors_algorithm).
* You can find the Jupyter Notebook for the classification task in [dictionary_creation/classification/01_knn.ipynb](dictionary_creation/classification/01_knn.ipynb).

This straightforward approach for classification is based on distances between vectors in a multidimensional feature space (in our case the word embeddings).
Simply put, the algorithm labels an instance based on the *k* nearest labeled neighbors of that instance, where *k* can take an arbitrary value.

Let us consider the following two-dimensional example (image taken from the Wikipedia article):

<p align="center">
  <img src="images/knn_example.png?raw=true" alt="KNN Example"/>
</p>

In this example, we have two differently labeled classes: the *blue squares* and the *red triangles*.
We have one unlabeled instance colored in green, which we want to assign to either of the two classes.
The resulting class for the green instance depends on how we set *k*.
Note that for a binary classification task (two classes), only odd numbers for *k* make sense as it could result in ties otherwise.

In the above example, we can observe the outcome for *k* = 3 (solid circle) and *k* = 5 (dashed circle).
For *k* = 3, the algorithm considers the three nearest neighbors which include one instance of the *blue squares* and two instances of the *red triangles*.
As the majority of neighbors are *red triangles*, it assigns our unlabeled instance to the class *red triangles*.
In the case of *k* = 5, the algorithm considers the five nearest neighbors, three of which are *blue squares* and two of which are *red triangles*.
Thus, it assigns our unlabeled instance to the class *blue squares*.

As you can see, the selection of *k* has a crucial impact on the result of the classification.
Further, there are additional hyperparameters for this classifier that affect the classification outcome.
For example, the distance measure to find nearest neighbors.
For word embeddings (e.g., TF-IDF, word2vec), we suggest to use cosine similarity because directions of vectors are more important than magnitudes of vectors.
To counteract the problem of the hyperparameter selection, one could conduct a grid search (similar to what we have done for the word2vec model) and evaluate the classified words with a test set containing a ground truth.
For our case, we decided to set *k* to 5 (a common value for *k*) and to evaluate the performance of our model as described in the following step.

#### Classifier Evaluation
* You can find the Jupyter Notebook for the classifier evaluation in [dictionary_creation/classification/02_evaluation.ipynb](dictionary_creation/classification/02_evaluation.ipynb).
* You can find our annotated and ready-to-use evaluation words for French, Italian and Spanish in [dictionary_creation/classification/ready_to_use/](dictionary_creation/classification/ready_to_use/).

To evaluate the performance of our classifier, we randomly extract a maximum of 1,000 words from each of the three sentiment classes and manually annotate them.
Note that our Jupyter Notebook provides a way to randomly extract words and to prepare *.csv* files for annotation.
For our dictionaries, we again let three independent annotators label the cumulated 3,000 words, respectively for each language (except for Spanish for which the classifier labeled only 649 positive and 440 negative words).
Similar to the annotation process for seed words, we only keep the annotated words for which at least two annotators agreed (majority vote).
We than compute balanced accuracy scores (between the labels from the classifier and the labels from annotators) to assess the prediction performance of our classifier and, thus, the quality of our computationally created dictionaries.
Here, we report:

Language | Balanced Accuracy
--- | --- 
French | 0.57 
Italian | 0.54
Spanish | 0.55

These values indicate that we outperform a random baseline (0.33) and are somewhat similar to other automated sentiment classification tasks in terms of performance (Mozetič et al. 2016).
However, note that we did not check all the words labeled by our classifiers but only a small and randomly drawn subset of them.

## Sentiment Analysis for the 18<sup>th</sup> Century

### Dictionaries
You can use our created sentiment dictionaries for French, Italian and Spanish for your own projects.
In particular, we provide three different forms of dictionaries for each of the three languages:
One dictionary containing all the manually annotated words used during seed word creation (see [Seed Words](#seed-words)) and evaluation of our classifiers (see [Classification](#classification)), one containing manually annotated seed words and the computationally extended words, as well as one containg manually annotated seed words and computatiopnally extended words that have been corrected using the manually annotated evaluation words.

#### Manually Annotated
* You can find manually annotated dictionaries for French, Italian and Spanish in [sentiment_analysis/dictionaries/manual/](sentiment_analysis/dictionaries/manual/).

These dictionaries contain all words that have been manually annotated during our dictionary creation process.
For each language, we provide a list of positive, neutral and negative words.
All words have been manually annotated by three experts familiar with the spectator periodicals.
To assess the agreement between annotators, we computed Fleiss' kappa (Fleiss 1971; [Wikipedia article](https://en.wikipedia.org/wiki/Fleiss%27_kappa)), respectively for seed word annotations as well as evaluation word annotations:

**Seed words:**
Language | kappa 
--- | --- 
French | 0.387 
Italian | 0.385 
Spanish | 0.370

**Classifier evaluation words:**
Language | kappa 
--- | --- 
French | 0.127 
Italian | 0.328 
Spanish | 0.300

These values suggest a fair agreement between annotators and reflect the typical discrepancies between humans regarding sentiment (Mozetič et al. 2016).
We only kept words for which at least two annotators agreed.

Number of words in the dictionaries:

Language | # positive | # negative | # neutral
--- | --- | --- | --- 
French | 1,045 | 1,071 | 3,150 
Italian | 1,789 | 1,196 | 2,696
Spanish | 681 | 798 | 3,529

#### Computationally Annotated
* You can find computationally created dictionaries for French, Italian and Spanish in [sentiment_analysis/dictionaries/computational/](sentiment_analysis/dictionaries/computational/).

These dictionaries contain manually annotated seed words as well as words for which we transferred sentiment through our classification (see [Classification](#classification)).
Note that these dictionaries are very specific to our text corpus (spectator periodicals).
If you are considering other data, you may want to use manually annotated words only or create extended dictionaries yourself (see [Creating Sentiment Dictionaries](#creating-sentiment-dictionaries)).

Number of words in the dictionaries:

Language | # positive | # negative | # neutral
--- | --- | --- | --- 
French | 4,713 | 2,350 | 17,499 
Italian | 4,365 | 1,652 | 25,494
Spanish | 1,034 | 691 | 19,070

#### Computationally Annotated and Corrected
* You can find computationally created and corrected dictionaries for French, Italian and Spanish in [sentiment_analysis/dictionaries/computational_corrected/](sentiment_analysis/dictionaries/computational_corrected/).

These dictionaries are very similar to the computationally extended ones, except that we corrected words based on manual annotations conducted for the classifier evaluations.
For example, if the KNN classifier labeled a word as positive but at least two annotators labeled it differently, we changed the word label to that of the annotators.

Number of words in the dictionaries:

Language | # positive | # negative | # neutral
--- | --- | --- | --- 
French | 4,216 | 2,272 | 18,074 
Italian | 4,387 | 1,674 | 25,450
Spanish | 692 | 812 | 19,291

### Computing sentiment

Using our created sentiment dictionaries, we compute the sentiment score *s* with

<p align="center">
  <img src="images/sentiment_formula.png?raw=true" alt="Sentiment Formula"/>
</p>

where *W<sub>p</sub>* is the number of positive words in a text and *W<sub>n</sub>* is the number of negative words in a text.
Thus, the sentiment score is a value ranging between −1 and +1, where values close to −1 are considered as negative, values close to +1 as positive, and where values close to zero indicate a neutral sentiment.
This formual is already implemented in our Jupyter Notebook.

### Data Preparation
* You can find the Jupyter Notebook for the data perparation in [sentiment_analysis/01_data_preparation.ipynb](sentiment_analysis/01_data_preparation.ipynb).

First, you need to prepare your textual data in order to work with our Jupyter Notebook.
Depending on the things you want to analyze, you may or may not provide additional attributes to your text files.
For example, you may state author names in order to analyze differences in sentiment across authors.
To add your custom attributes, you need to append them before your text in the *.txt* files.
Our data preparation Notebook than translates your individual text files into a pandas DataFrame, which makes it easier for you to work with your texts in Python.
Please note that you have to use the following file format:

* Each attribute must be provided in an individual line.
* Provide the name of the attribute in the beginning of a line followed by an `=` and then the value for the attribute. Please do not include spaces between text and the `=`. For example: `year=1786`
* You can provide any attribute you like. Just make sure that attribute names are unique.
* Attribute names can include spaces. For example: `periodical title=La Spectatrice`.
* The actual text must be provided at last following a `text=`.
* You can use line breaks in the text, just not for attributes.

The following snippet serves as an exemplary input *.txt* file:

```
year=1711
author=Justus Van Effen
text=Si je prends la liberté de vous dédier cet Ouvrage; ce n’est en aucune maniere pour me ménager une favorable occasion d’instruire les hommes de votre mérite, & de vous donner, même avec sobrieté, les éloges dont vous êtes digne...
```

If you adhere to this file format, our notebook should create a pickled pandas DataFrame which you can use in the subsequent sentiment analysis Notebook.
Note that the inclusion of additional attributes is optional.
If you do not want to provide attributes, just start your input text files with `text=`. For example:

```
text=Si je prends la liberté de vous dédier cet Ouvrage; ce n’est en aucune maniere pour me ménager une favorable occasion d’instruire les hommes de votre mérite, & de vous donner, même avec sobrieté, les éloges dont vous êtes digne...
```

### Example Texts
* You can find zipped example texts for French, Italian and Spanish in [sentiment_analysis/example_texts.zip](sentiment_analysis/example_texts.zip).

We provide example texts for French, Italian and Spanish so you can see how you should prepare your text files and try the different analysis methods.
Just download and unzip `example_texts.zip` in the directory `sentiment_analysis/`.
After doing so, the directory should look like this:

```
sentiment_analysis/dictionaries/
sentiment_analysis/example_texts/
sentiment_analysis/01_data_preparation.ipynb
sentiment_analysis/02_sentiment_analysis.ipynb
```

You can now use `01_data_preparation.ipynb` to generate a pandas DataFrame and then use `02_sentiment_analysis.ipynb` to try the sentiment analysis methods with our example texts.

### Analysis
* You can find the Jupyter Notebook for the sentiment analysis in [sentiment_analysis/02_sentiment_analysis.ipynb](sentiment_analysis/02_sentiment_analysis.ipynb).

In this Notebook, we provide multiple ways to analyze sentiment of your texts.
For example, you can analyze how sentiment varies across the attributes you defined.
Just download and open the Notebook to see further analysis methods.

#### Examples
Here we show some examples of what you can do with our Jupyter Notebook.

**Bar plot:**
<p align="center">
  <img src="images/bar_plot_example.png?raw=true" alt="Bar plot"/>
</p>

This bar plot shows the mean sentiment over attributes (in this case for two authors).

**Box plot:**
* You can read more about box plots in the corresponding [Wikipedia article](https://en.wikipedia.org/wiki/Box_plot).

<p align="center">
  <img src="images/box_plot_example.png?raw=true" alt="Box plot"/>
</p>

The box plot is a convenient way to learn more about the distribution of the underlying data.
In this example, we show individual box plots for each year.
The horizontal green lines indicate medians, while blue lines indicate the first and third quartile.
Whiskers (horizontal black lines) indicate minimum and maximum values still within 1.5 interquartile ranges.

**Line plot:**
<p align="center">
  <img src="images/line_plot_example.png?raw=true" alt="Line plot"/>
</p>

A line plot suits perfectly to plot the continuous development of sentiment. In this example, we plot the mean sentiment over individual years.

**Sentiment words highlighting:**
<p align="center">
  <img src="images/word_highlight_example.png?raw=true" alt="Highlighting plot"/>
</p>

We provide a way to highlight the words conveying either a positive (colored in green) or negative sentiment (colored in red) in your texts.

**Sentiment development in a file:**
<p align="center">
  <img src="images/sentiment_development_example.png?raw=true" alt="Sentiment development example"/>
</p>

We show how you can analyze the sentiment development in an individual text file by splitting it into separate chunks, count the number of positive and negative words in chunks and compute respective sentiments.
This is useful if you want to see how sentiment progresses, for example, in a story.

**Hypothesis tests:**
* You can read more about statistical hypothesis testing in the corresponding [Wikipedia article](https://en.wikipedia.org/wiki/Statistical_hypothesis_testing).

We can further interpret our data by making assumptions about it.
Typically, we ask questions about our data, for example, whether two samples are drawn from the same distribution or not.
Statistical hypothesis tests provide a likelihood and confidence about the answers to the respective questions and allow us to confirm or reject our assumptions.
Our Jupyter Notebook contains an example of a statistical hypothesis test to give you an idea of how to conduct them with Python.

### Where to Go From Here
Once you have computed sentiment, it is time to interpret your results. While sentiment analysis is great to extend the knowledge about your data as well as to provide an overview about expressed opinions, results based on machine methods should always be considered with caution.

Our tool chain can be easily adapted and serves as a foundation for future projects.
For example, one could implement a more sophisticated algorithm that considers the context of words, such as negations that could affect the sentiment orientation. Further, at current state, we only consider positive and negative words to compute sentiment, but our tool chain also creates lists of neutral sentiment words. These words could also be considered in the sentiment formula.

Our specific way of selecting seed words has also a significant impact on the computationally created dictionaries. You could try different seed words and see how this affects your dictionaries. Additionally, you could try other classification methods to transfer sentiment from seed words to other words in a similar context.

Finally, it may be interesting to compare sentiment dictionaries for the 18<sup>th</sup> century with sentiment dictionaries for modern times.

## Notes

### Citation Information
Please reference our work if you used our dictionaries or tool chain:

```
Koncar, P., Druml, L., Ertler, K.-D., Fuchs, A., Geiger, B. C., Glatz, C., Helic, D., Hobisch, E., Mayer, P., Saric, S., Scholger, M. & Voelkl, Y. (2021) A Sentiment Tool Chain for Languages of the 18th Century. https://github.com/philkon/sentiment-tool-chain
```

### The DiSpecs Project
The idea for and the development of this tool chain is a result of the **DiSpecs** (DISTANT SPECTATORS: DISTANT READING FOR PERIODICALS OF THE ENLIGHTENMENT) project.
Please visit [our project website](https://gams.uni-graz.at/dispecs) if want to learn more about our work on Spectator periodicals.

### Development and Repository
* Philipp Koncar

### Word Annotations
We want to thank our annotators (in alphabetical order) for their efforts in providing high quality sentiment annotations for the 18<sup>th</sup> century:
* Lena Druml
* Klaus-Dieter Ertler
* Alexandra Fuchs
* Christina Glatz
* Elisabeth Hobisch
* Pia Mayer
* Yvonne Völkl

### Acknowledgments
This work was funded by CLARIAH-AT and partly funded by the go!digital programme of the Austrian Academy of Sciences.

We want to thank the following persons (in alphabetical order) for their fruitful input and without whom this work would not have been possible:
* Bernhard C. Geiger
* Denis Helic
* Thorsten Ruprechter
* Sanja Saric
* Martina Scholger

Further, we thank *Tinghui Duan* for reporting bugs to us.

## References

* Ertler, K-D, Fuchs A, Fischer M, Hobisch E, Scholger M, Völkl Y (Unknown Month 2011) The Spectators in the international context. https://gams.uni-graz.at/spectators. Accessed 16 Feb 2021.

* Fleiss, J. L. (1971). Measuring nominal scale agreement among many raters. Psychological bulletin, 76(5), 378.

* Freitas, A., Barzegar, S., Sales, J. E., Handschuh, S., & Davis, B. (2016, November). Semantic relatedness for all (languages): A comparative analysis of multilingual semantic relatedness using machine translation. In European Knowledge Acquisition Workshop (pp. 212-222). Springer, Cham.

* Hutto, CJ, Gilbert E (2014) Vader: A parsimonious rule-based model for sentiment analysis of social media text In: Eighth International AAAI Conference on Weblogs and Social Media. The AAAI Press, California.

* Maas, AL, Daly RE, Pham PT, Huang D, Ng AY, Potts C (2011) Learning word vectors for sentiment analysis In: Proceedings of the 49th Annual Meeting of the Association for Computational Linguistics: Human Language Technologies-volume 1, 142–150.. Association for Computational Linguistics, Stroudsburg.

* Mikolov, T, Chen, K, Corrado, G, & Dean, J (2013) Efficient estimation of word representations in vector space. arXiv preprint arXiv:1301.3781.

* Mozetič, I., Grčar, M., & Smailović, J. (2016). Multilingual Twitter sentiment classification: The role of human annotators. PloS one, 11(5), e0155036.

* Pang, B, Lee L, Vaithyanathan S (2002) Thumbs up?: sentiment classification using machine learning techniques In: Proceedings of the ACL-02 Conference on Empirical Methods in Natural Language processing-Volume 10, 79–86.. Association for Computational Linguistics, Pennsylvania.

* Pang, B, Lee L, et al (2008) Opinion mining and sentiment analysis. Found Trends® Inf Retr 2(1–2):1–135.

* Scholger, M (2018) “Spectators” in the International Context - A Digital Scholarly Edition In: Discourses on Economy in the Spectators, 229–247.. Verlag Dr. Kovac, Hamburg.

* Thelwall, M, Buckley K, Paltoglou G, Cai D, Kappas A (2010) Sentiment strength detection in short informal text. J Am Soc Inf Sci Technol 61(12):2544–2558.

* Ye, Q, Zhang Z, Law R (2009) Sentiment classification of online reviews to travel destinations by supervised machine learning approaches. Expert Syst Appl 36(3):6527–6535.

* Zhang, L, Wang S, Liu B (2018) Deep learning for sentiment analysis: A survey. Wiley Interdiscip Rev Data Min Knowl Disc 8:1253.
