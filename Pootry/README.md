# Pootry

A pootry.. no-uh, I mean, poetry generator.

### Description

`Pootry` is an exploratory `Natural Language Processing` project about `Text Generation` with the focus of `Poem Creator`. As such, the built model is trained to make a set of verses defining what a poem is from a `seed text` given as input from the user. The result might turn out to be weird as it is not made for production.

`Pootry` uses `Deep Learning` model mainly with the help of `Tensorflow Keras` library, also preprocessed with the help of `Regular Expression` and `Numpy` for tensor-like operations. The model is also made with self-trained `Words Embeddings` with `Global Vector` model by Stanford University. The GloVe mddel used is the Python version one and using the [actual version](https://nlp.stanford.edu/projects/glove/) might produce better matrix.

The main model used in the project is `Bidirectional Long Short-Term Memory` or `LSTM` with `Embedding` and `Dropout` layer, each for embedding the embeddings matrix and to avoid overfitting. Parameters used for each layers can be changed freely in the source code. The model will take in `Ragged Tensors` from preprocessed tokens sequences with the `Tokenizer`.

The `.txt` data can be found in the `\data` folder. The data `poems.txt` can be found [here](https://github.com/Nwosu-Ihueze/poem_generator), while the data `shakespeare.txt` can be found [here](https://dataskat.s3.eu-west-3.amazonaws.com/data/Shakespeare_alllines.txt). Do take attention with the `shakespeare` data due to its large size.

### Guide

-   Simply use the Notebook and run all the cells sequentially either with Jupyter Notebook or Google Colab (recommended). The notebook will download some external libraries.
-   **WARNING**: The training takes so much time `:]`. It takes approximately 25 minutes and not even 10 epochs are done.

### Creator

-   Richard Rivaldo / 13519185

### Possible Improvements

-   Better preprocessing and cleaning part
-   Better sequence modelling and tokenization
-   Hyperparameters tuning and layer choices
-   More and Bigger Embeddings Matrix with GloVe (or maybe other word embeddings model?)
-   Was curious and tried to apply Attention into the model, but errors came and dropped many frame bombs. :)))

### References

-   [A Comprehensive Python Implementation of GloVe](https://towardsdatascience.com/a-comprehensive-python-implementation-of-glove-c94257c2813d)
-   [Creating a Poems Generator using Word Embeddings](https://towardsdatascience.com/creating-a-poems-generator-using-word-embeddings-bcc43248de4f)
-   [Generating Haiku with Deep Learning (Part 1)](https://towardsdatascience.com/generating-haiku-with-deep-learning-dbf5d18b4246)
-   [Generative poetry with LSTM](https://towardsdatascience.com/generative-poetry-with-lstm-2ef7b63d35af)
-   [Getting Started with Word2Vec and GloVe in Python](https://sites.google.com/site/nttrungmtwiki/home/it/data-mining/text-mining/nlp/getting-started-with-word2vec-and-glove-in-python)
-   [GloVe: Global Vectors for Word Representation](https://nlp.stanford.edu/projects/glove/)
-   [GRAMPS: Generating Really Awesome Metaphorical Poetry (Sometimes)](https://github.com/Tahlor/GRAMPS)
-   [How to Calculate Precision, Recall, F1, and More for Deep Learning Models](https://machinelearningmastery.com/how-to-calculate-precision-recall-f1-and-more-for-deep-learning-models/)
-   [How to Choose Loss Functions When Training Deep Learning Neural Networks](https://machinelearningmastery.com/how-to-choose-loss-functions-when-training-deep-learning-neural-networks/)
-   [How to Use Word Embedding Layers for Deep Learning with Keras](https://machinelearningmastery.com/use-word-embedding-layers-deep-learning-keras/)
-   [Intuitive Guide to Understanding GloVe Embeddings](https://towardsdatascience.com/light-on-math-ml-intuitive-guide-to-understanding-glove-embeddings-b13b4f19c010)
-   [Module: tf](https://www.tensorflow.org/api_docs/python/tf)
-   [My poetry generator passed the Turing Test](https://rpiai.com/other/poetry/)
-   [Poem Generator with LSTM](https://medium.com/analytics-vidhya/poem-generator-with-lstm-29135483d588)
-   [Poetry Generation using Seq2Seq](https://www.kaggle.com/pikkupr/poemgeneration-using-seq2seq-memory-networks)
-   [Ragged tensors](https://www.tensorflow.org/guide/ragged_tensor)
-   [Text classification with RaggedTensors and Tensorflow Text](https://dzlab.github.io/nlp/2019/12/08/tensorflow-text-imdb/)
-   [Text generation with an RNN](https://www.tensorflow.org/text/tutorials/text_generation)
-   [Text Generation With LSTM Recurrent Neural Networks in Python with Keras](https://machinelearningmastery.com/text-generation-lstm-recurrent-neural-networks-python-keras/)
-   [Text Generation Using Pre-trained Word Embedding Layer](https://renom.jp/notebooks/tutorial/time_series/text_generation_using_pretrained_word_embedding_layer/notebook.html)
-   [Training a seq2seq on Modern Poetry with Softmax Temperature](https://github.com/hollygrimm/seq2seq-poetry)
-   [Train Your First Embedding Models](https://openclassrooms.com/en/courses/6532301-introduction-to-natural-language-processing/7132231-train-your-first-embedding-models)
-   [Using pre-trained word embeddings](https://keras.io/examples/nlp/pretrained_word_embeddings/)
-   [Using TensorFlow Ragged Tensors](https://towardsdatascience.com/using-tensorflow-ragged-tensors-2af07849a7bd)
