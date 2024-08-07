# Text Style Classification between _Dante, Italian and Neapolitan language_: a comparison between CNN, RNNs and Transformer based Classifiers 📖 🤔
A Pytorch implementation of Text Style Classification. 

The goal of this project is to compare different architecture accuracy in predicting 3 italic language styles: _Dante, Italian and Neapolitan_. 

The Classifiers taken in consideration are CNN, RNNs, and Transformer based:
* CNN Classifier: made up of three 2D Convolutional layers with a 3x3 kernel
* RNN Classifier: takes the last hidden state of a RNN and classifies sentences from it
* GRU Classifier: takes the last hidden state of a RNN with GRU cell and classifies sentences from it
* LSTM Classifier: takes the last hidden state of a RNN with LSTM cell and classifies sentences from it
* Transformer Classifier: averages the output of a Transformer Encoder over words in a sentence. It's based of self-attention, as described in . . .

The text corpus taken in consideration for the three language are:
* Dante: _Divina Commedia_
* Italian: _Uno, nessuno e centomila_ by Luigi Pirandello, _I Malavoglia_ by Giovanni Verga
* Neapolitan: _Lo cunto de li cunti_ by Giambattista Basile

# Table of Contents
- [Structure](#Structure)
- [Requirements](#Requirements)
- [Usage](#Usage)
- [Results](#Results)
- [Bonus](#Bonus)

# Structure
* [`PRE_TRAINED_TEXT_CLASSIFIER.ipynb`](PRE_TRAINED_TEXT_CLASSIFIER.ipynb) is the main notebook containing pre trained models, classification examples and accuracy computations
* [`TEXT_CLASSIFIER.ipynb`](TEXT_CLASSIFIER.ipynb) is a notebook where all the models are defined and trained, with classification examples and accuracy computations
* [`models.py`](models.py) is the module with model class definitions
* [`data_config.py`](data_config.py) is the module containing the functions for making the dataset in form of torch Dataloaders
* [`training_function.py`](training_function.py) is the module containing the functions for training
* [`accuracy.py`](accuracy.py) is the module containing the function computing model accuracy
* [`text_corpus`](text_corpus) repository contains the three corpus used for training
* [`pretrained`](pretrained) repository contains the .pth files with the parameters from pre-trained models

# Requirements
- Numpy
- Matplotlib
- Pytorch
  
# Usage
First clone the repository:

```bash
git clone git@github.com:MassimoMario/Italic_Text_Style_Classification.git
```

Make sure to have Pytorch installed:
```bash
pip install torch
```

Run cells from [`TEXT_CLASSIFIER.ipynb`](TEXT_CLASSIFIER.ipynb) notebook if you want to train the models yourself, or run [`PRE_TRAINED_TEXT_CLASSIFIER.ipynb`](PRE_TRAINED_TEXT_CLASSIFIER.ipynb) for using pre-trained models. Both notebooks have classification examples  and accuracy computation after every model section.
# Results
Here the training curves for these five classifiers:


Table with prediction accuracies evaluated on test datasets with 1017 sentences for each style:

| Classifier | _Dante_ | _Italian_ | _Neapolitan_ | Overall |
| --- | --- | --- | --- | --- |
| CNN | 98.52% | 99.21% | 99.80% | 99.17% |
| RNN | 98.23% | 97.64% | 97.80% | 97.89% |
| GRU | 97.54% | 99.21% | 99.60% | 98.78% |
| LSTM | 98.32% | 99.50% | 99.60% | 99.14% |
| Transformer | 98.52% | 99.41% | 100% | 99.31% |


# Bonus
Since when we italians are young, we learn in school that _Inferno, Purgatorio_ and _Paradiso_, the three main parts of _Divina Commedia_, have been written in 3 different styles.

Can these models capture these stylistic differences? 

| Classifier | _Inferno_ | _Purgatorio_ | _Paradiso_ | Overall |
| --- | --- | --- | --- | --- |
| CNN | 57.01% | 20.82% | 70.65% | 49.49% |
| RNN | 52.33% | 24.04% | 38.32% | 38.23% |
| GRU | 47.66% | 28.16% | 50.00% | 41.93% |
| LSTM | 59.35% | 25.51% | 63.17% | 49.34% |
| Transformer | 48.53% | 39.88% | 60.77% | 49.73% |

From these accuracies it seems they can't  😞

Even if on _Inferno_ and _Paradiso_ test set models perform much better respect to predicting _Purgatorio_ test set. I guess it's because _Inferno_ and _Paradiso_ have a more recognizable writing style given by a precise choice of words by Dante.
