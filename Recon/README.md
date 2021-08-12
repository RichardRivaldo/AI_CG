# Recon

A `Text Recognition` project with no Tesseract involved :D

### Description

`Recon` is a `text recognition` model implemented with `Deep Learning` architecture beneath it. `Text Recognition` is the problem of analyzing and extracting texts from an image, and for the case of something like `Handwritten Recognition` or `Synthetic Texts`, it can be classified as an `Optical Character Recognition` problem.

Currently, there are many approaches that one can take to create the model to do the task, such as Google's infamous Tesseract library which is known for its great accuracy despite being quite simple in its abstraction.

For the sake of diving deeper into deep learning, Recon is using the sophisticated `CRNN` or `Convolutional Recurrent Neural Network` architecture. Basically, this architecture is a hybrid of both `CNN` or `Convolutional Neural Network` and `RNN` or `Recurrent Neural Network`.

CNN works closely with images and consists of `downsampling` blocks through `convoluting` the images and `pooling` them so that the model knows which features are needed to be extracted. While CNN works with images, RNN works with recursive sequences data like texts, and that is totally compatible with this task. The RNN model used in Recon is `LSTM` or `Long Short-Term Memory`. Since the data is in form of images, then the architecture will feed the data to the CNN first. After CNN processed the images, then RNN will take turns to process it.

For its loss, the model will use `CTC` or `Connectionist Temporal Classification` loss mechanism. The mechanism forces the model to put and fit all characters in an image to its timesteps. To handle words with duplicated characters sticking together, CTC uses a special character, namely `Blank Token`, to separate them both. Not only as a loss, CTC can also be used as a `decoder` by merging all the characters in each timesteps and removing all blanks and duplicated characters that doesn't have any blanks between them.

### Guide

-   Simply use the Notebook and run all the cells sequentially either with Jupyter Notebook or Google Colab (recommended). The notebook will be expecting to get data from Google Drive in form of zipped images.
-   ~~**WARNING**: The training takes so much time.~~ I TAKE THAT BACK. I totally forgot that Colab has GPU Runtime settings and it is satisfying to see how fast the training is with many attempts on changing all available parameters. Thanks and love you, Colab! :D
-   The images dataset is mounted from my own drive. The full dataset used for the project can be found [here](https://drive.google.com/drive/folders/1j_kdJ4VO0n1DSJdnXOmFJZm4oTn5eyC7?usp=sharing). Only a subset of images with lowercase characters are used in this project to make it similar to the initial dataset provided. :)
-   I'm not using the provided data because I tried to train the model with it and the results are actually bad.. Anyway, the saved model for the last Notebook result is also in the link provided above. Hopefully, I downloaded the correct file since I saved and trained the model so many times...

### Creator

-   Richard Rivaldo / 13519185

### Possible Improvements

-   Hyperparameter tunings
-   Better cleaning and preprocessing
-   More data would surely bring better results
-   Planned to try to put `Attention!` in the arch, but let's see!

### References

-   [A-CRNN-model-for-Text-Recognition-in-Keras](https://github.com/TheAILearner/A-CRNN-model-for-Text-Recognition-in-Keras)
-   [A gentle introduction to OCR](https://towardsdatascience.com/a-gentle-introduction-to-ocr-ee1469a201aa)
-   [Building Custom Deep Learning Based OCR models](https://nanonets.com/blog/attention-ocr-for-text-recogntion/)
-   [Creating a CRNN model to recognize text in an image (Part-1)](https://theailearner.com/2019/05/29/creating-a-crnn-model-to-recognize-text-in-an-image-part-1/)
-   [Creating a CRNN model to recognize text in an image (Part-2)](https://theailearner.com/2019/05/29/creating-a-crnn-model-to-recognize-text-in-an-image-part-2/)
-   [CRNN_Attention_OCR_Chinese](https://github.com/wushilian/CRNN_Attention_OCR_Chinese)
-   [Get started with deep learning OCR](https://towardsdatascience.com/get-started-with-deep-learning-ocr-136ac645db1d)
-   [How did I write an own OCR program using Keras and TensorFlow in Python](https://towardsdatascience.com/how-did-i-train-an-ocr-model-using-keras-and-tensorflow-7e10b241c22b)
-   [How to automatically deskew (straighten) a text image using OpenCV](https://becominghuman.ai/how-to-automatically-deskew-straighten-a-text-image-using-opencv-a0c30aed83df)
-   [How to implement ctc loss using tensorflow keras (feat. CRNN example)](https://chadrick-kwag.net/tf-keras-rnn-ctc-example/)
-   [How to train a Keras model to recognize text with variable length](https://www.dlology.com/blog/how-to-train-a-keras-model-to-recognize-variable-length-text/)
-   [Image Pre-processing](https://towardsdatascience.com/image-pre-processing-c1aec0be3edf)
-   [Improve Accuracy of OCR using Image Preprocessing](https://medium.com/cashify-engineering/improve-accuracy-of-ocr-using-image-preprocessing-8df29ec3a033)
-   [Latest Deep Learning OCR with Keras and Supervisely in 15 minutes](https://hackernoon.com/latest-deep-learning-ocr-with-keras-and-supervisely-in-15-minutes-34aecd630ed8)
-   [OCR model for reading Captchas](https://keras.io/examples/vision/captcha_ocr/)
-   [OCR with Deep Learning: The Curious Machine Learning Case](https://labelyourdata.com/articles/ocr-with-deep-learning/)
-   [OCR with Keras, TensorFlow, and Deep Learning](https://www.pyimagesearch.com/2020/08/17/ocr-with-keras-tensorflow-and-deep-learning/)
-   [Optical Character Recognition Using Deep Learning Techniques](https://heartbeat.fritz.ai/optical-character-recognition-using-deep-learning-techniques-1376605b022a)
-   [Pre-Processing in OCR!!!](https://towardsdatascience.com/pre-processing-in-ocr-fc231c6035a7)
-   [Text Recognition With CRNN-CTC Network](https://wandb.ai/authors/text-recognition-crnn-ctc/reports/Text-Recognition-With-CRNN-CTC-Network--VmlldzoxNTI5NDI)
-   [Understanding of CRNN text recognition algorithm](https://www.programmersought.com/article/27554772461/)
