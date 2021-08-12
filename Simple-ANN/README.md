# Simple-ANN

Implementation of Artificial Neural Network architecture

### Description

`Deep Learning` can be said as the current state-of-art of Machine Learning and Artificial Intelligence. It is a subfield of Machine Learning with the concern of mimicking how human's brain works in learning a data and applying it to make a prediction on a new unseen-before data.

Inspired by the neuron of the brain, most models in this subfield is named with `Neural Network`. One of the simplest model implemented in this repository is called `Artificial Neural Network`. As the name suggests, the model consists of `neurons` connected to each others, sharing and getting informations through the networks built between them.

The neurons can then be stacked to each other, and the stack will be called `layers`. Each layers will be connected to the other layers. Generally, most models will have one `input layer` and one `output layer` as a pair. Between them, there will be hidden layers to process the input given from the input layer thoroughly.

In this model, there will be three main algorithms applied iteratively while fitting and training the model. They are `Forward Propagation`, `Backward Propagation`, and `Weight Updating`. `Forward Propagation` is about propagating the input forward from the input layer through the `Activation Function`.

Conversely, `Backward Propagation` uses all values obtained in the step before and tries to calculate the slope or gradient by comparing real values from the dataset with values from `Forward Propagation`. Lastly, the model will use the gradient to descend and minimize the `Loss` by adjusting `weights` and `biases` in every neurons, and such will be called `Gradient Descent`.

The model in this implementation will consist of only one hidden layer. The number of neurons in each layers are pre-determined in the algorithm. The values that can be adjusted in the algorithm are `Learning Rate`, `Epochs`, and `Batch Size` used in the `Stochastic Gradient Descent` optimization.

### Guide

-   Go to the home directory of the project through `Terminal` or `Command Prompt` with the `cd <Simple-ANN_directory>`. Don't get too deep to the `src` folder!
-   Enter the command needed to run an algorithm. The example of the command will be given below.
    -   `python src/main.py --dataset data\heart.csv`
    -   `python src/main.py --dataset data\heart.csv --lr 0.9`
    -   `python src/main.py --dataset data\heart.csv --batch_size 2 --epochs 2000`
-   The compulsory argument is only `--dataset`.
-   The rest of the arguments can be freely adjusted. If not given, then the default value for each arguments will be used instead.
-   If the data wants to be changed, put the data in the `data` folder. One thing to note is there isn't any preprocessing part of the data. All values in the sample data is numerical.
-   The model will be automatically trained and after the training is done, the model will pick a random data index from the `Test Set` and predict the value.
-   The main output of the prediction will be: `Input Data`, `Expected Label`, and `Predicted Label`. The `MSE Loss Function` will also be shown for each epoch.

### Creator

-   Richard Rivaldo / 13519185

### Possible Improvements

-   Better accuracy
-   More hidden layers
-   Better algorithm

### References

-   [All the Backpropagation derivatives](https://medium.com/@pdquant/all-the-backpropagation-derivatives-d5275f727f60)
-   [An overview of gradient descent optimization algorithms](https://ruder.io/optimizing-gradient-descent/index.html#shufflingandcurriculumlearning)
-   [Build an Artificial Neural Network From Scratch: Part 1](https://www.kdnuggets.com/2019/11/build-artificial-neural-network-scratch-part-1.html)
-   [Building a Neural Network From Scratch Using Python (Part 1)](https://heartbeat.fritz.ai/building-a-neural-network-from-scratch-using-python-part-1-6d399df8d432)
-   [Building a Neural Network From Scratch Using Python (Part 2)](https://heartbeat.fritz.ai/building-a-neural-network-from-scratch-using-python-part-2-testing-the-network-c1f0c1c9cbb0)
-   [Building Neural Network from scratch](https://towardsdatascience.com/building-neural-network-from-scratch-9c88535bf8e9)
-   [Gradient Descent in Python: Implementation and Theory](https://stackabuse.com/gradient-descent-in-python-implementation-and-theory)
-   [Gradient Descent From Scratch](https://towardsdatascience.com/gradient-descent-from-scratch-e8b75fa986cc)
-   [How to Code a Neural Network with Backpropagation In Python (from scratch)](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/)
-   [Neural Networks from Scratch](https://github.com/nishanthballal-9/Neural-Networks-from-scratch)
-   [Programming a neural network from scratch](https://www.ritchievink.com/blog/2017/07/10/programming-a-neural-network-from-scratch/)
-   [Python AI: How to Build a Neural Network & Make Predictions](https://realpython.com/python-ai-neural-network/)
-   [Scratch Implementation of Stochastic Gradient Descent using Python](http://binaryplanet.org/2020/04/scratch-implementation-of-stochastic-gradient-descent-using-python/)
-   [The Ultimate Beginner’s Guide To Implement A Neural Network From Scratch](https://towardsdatascience.com/the-ultimate-beginners-guide-to-implement-a-neural-network-from-scratch-cf7d52d91e00)
-   [Understanding and coding Neural Networks From Scratch in Python and R](https://www.analyticsvidhya.com/blog/2020/07/neural-networks-from-scratch-in-python-and-r/?#)
-   [What are artificial neural networks (ANN)?](https://bdtechtalks.com/2019/08/05/what-is-artificial-neural-network-ann/)
-   [What is Deep Learning?](https://machinelearningmastery.com/what-is-deep-learning/)
