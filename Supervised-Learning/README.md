# Supervised-Learning

K-Nearest Neighbor, Logistic Regression, and Iterative Dichotomiser 3 Algorithm

### Description

`Supervised Learning` is one of the most popular type of Machine Learning algorithm. It consists of many different algorithm, each with their own advantages and disadvantages, comparatively to others.

Supervised Learning can be used for both `classification` and `regression`. Among all algorithms, there are three algorithms used for classification problem. They are `K-Nearest Neighbors`, `Logistic Regression`, and `Iterative Dichotomiser 3`.

`K-Nearest Neighbors` is the algorithm based on finding nearest data and find the `majority vote` to determine which category a predicted data falls in. The metric for distance used to compare each data can be `Euclidean Distance` or `Manhattan Distance` for numerical features, or `Hamming Distance` for categorical features.

`Logistic Regression`, on the other hand, is similar to `Linear Regression` for `Regression`. The thing that differs them is that `Logistic Regression` uses complex mathematical function such as `Sigmoid Function` and `Gradient Descent` and the output will be categorical instead of numerical values.

Lastly, `Iterative Dichotomiser 3` is an algorithm to generate decision tree, a tree made to classify a data based on certain rules. This algorithm uses `Information Gain` calculation and `Entropy` value of the information in the data to determine its importance in determining the data label.

The dataset used in this project is a `Churn Modelling` dataset. It contains several categorical and numerical features, and only has one binary label to predict the churn value of a customer. The dataset still needs to be preprocessed before feeding it to the models. The dataset is taken from Machine Learning A-Z Course on Udemy.

### Guide

-   Go to the home directory of the project through `Terminal` or `Command Prompt` with the `cd <Supervised-Learning_directory>`. Don't get too deep to the `src` folder!
-   Enter the command needed to run an algorithm. The example of the command will be given below.
    -   `K-Nearest Neighbors`: `python src\main.py --knn --data data\dataset.csv --k_nearest 3`
    -   `Logistic Regression`: `python src\main.py --log_reg --data data\dataset.csv --lr 0.3 --epochs 1000`
    -   `Iterative Dichotomiser 3`: `python src\main.py --id_3 --data data\dataset.csv`
-   Keep in mind that the first argument (aside from file name) should be the type of the algorithm!
-   All other values can be modified.
-   If the data wants to be changed, put the data in the `data` folder. One thing to note is the preprocessing part of the data needs to be changed too. The corresponding file can be found in `src/preprocess.py`.
-   The model will be automatically trained and after the training is done, the model will pick a random data index from the `Test Set` and predict the value.
-   The main output of the prediction will be: `Input Data`, `Expected Churn`, and `Predicted Churn`.

### Creator

-   Richard Rivaldo / 13519185

### Possible Improvements

-   Better accuracy + other properties (e.g. overfitting handler)
-   Better time and space complexity
-   More general functionalities and pipelines

### References

-   K-Nearest Neighbor

    -   [Develop k-Nearest Neighbors in Python From Scratch](https://machinelearningmastery.com/tutorial-to-implement-k-nearest-neighbors-in-python-from-scratch/)
    -   [Implementation of K Nearest Neighbors](https://www.geeksforgeeks.org/implementation-k-nearest-neighbors/)
    -   [Implementing Your Own k-Nearest Neighbor Algorithm Using Python](https://www.kdnuggets.com/2016/01/implementing-your-own-knn-using-python.html/3)
    -   [K-Nearest Neighbors Algorithm in Python and Scikit-Learn](https://stackabuse.com/k-nearest-neighbors-algorithm-in-python-and-scikit-learn)
    -   [Summary of KNN algorithm when used for classification](https://medium.com/analytics-vidhya/summary-of-knn-algorithm-when-used-for-classification-4934a1040983)
    -   [The k-Nearest Neighbors (kNN) Algorithm in Python](https://realpython.com/knn-python/#drawbacks-of-knn)

-   Logistic Regression

    -   [Building a Logistic Regression in Python](https://towardsdatascience.com/building-a-logistic-regression-in-python-301d27367c24)
    -   [Cost, Activation, Loss Function|| Neural Network|| Deep Learning. What are these?](https://medium.com/@zeeshanmulla/cost-activation-loss-function-neural-network-deep-learning-what-are-these-91167825a4de)
    -   [How To Implement Logistic Regression From Scratch in PythonHow To Implement Logistic Regression From Scratch in Python](https://machinelearningmastery.com/implement-logistic-regression-stochastic-gradient-descent-scratch-python/)
    -   [How to convert deep learning gradient descent equation into python](https://stackoverflow.com/questions/45832369/how-to-convert-deep-learning-gradient-descent-equation-into-python)
    -   [Implementation of Logistic Regression from Scratch using Python](https://www.geeksforgeeks.org/implementation-of-logistic-regression-from-scratch-using-python/)
    -   [Logistic Regression from Scratch](https://towardsdatascience.com/logistic-regression-from-scratch-69db4f587e17)
    -   [Logistic Regression From Scratch in Python](https://towardsdatascience.com/logistic-regression-from-scratch-in-python-ec66603592e2)
    -   [Logistic Regression From Scratch in Python [Algorithm Explained]](https://www.askpython.com/python/examples/logistic-regression-from-scratch)
    -   [Logistic Regression Machine Learning Algorithm in Python from Scratch](https://dhirajkumarblog.medium.com/logistic-regression-in-python-from-scratch-5b901d72d68e)

-   Iterative Dichotomiser 3

    -   [Building a ID3 Decision Tree Classifier with Python](https://guillermoarriadevoe.com/blog/building-a-id3-decision-tree-classifier-with-python)
    -   [Decision Tree - Classification](https://www.saedsayad.com/decision_tree.htm)
    -   [Decision Trees: ID3 Algorithm Explained](https://towardsdatascience.com/decision-trees-for-classification-id3-algorithm-explained-89df76e72df1)
    -   [Decision Tree Introduction with example](https://www.geeksforgeeks.org/decision-tree-introduction-example/)
    -   [Design and Implementation of Iterative Dichotomiser 3 (ID3) Decision Tree Classifier with Python](<https://github.com/samialperen/id3_decision_tree_classifier/blob/master/doc/Design%20and%20Implementation%20of%20Iterative%0ADichotomiser%203%20(ID3)%20Decision%20Tree%20Classifier%20with%0APython.pdf>)
    -   [ID3 Algorithm](http://athena.ecs.csus.edu/~mei/177/ID3_Algorithm.pdf)
    -   [ID3 (Iterative Dichotomiser 3) Algorithm - How does it work?](https://community.dataquest.io/t/id3-iterative-dichotomiser-3-algorithm-how-does-it-work/3113)
    -   [Iterative Dichotomiser 3 (ID3) Algorithm From Scratch](https://automaticaddison.com/iterative-dichotomiser-3-id3-algorithm-from-scratch/)
    -   [Iterative Dichotomiser 3 (ID3) algorithm – Decision Trees – Machine Learning](https://mariuszprzydatek.com/2014/11/11/iterative-dichotomiser-3-id3-algorithm-decision-trees-machine-learning/)
    -   [Understanding of ID3 algorithm and its advantages and disadvantages](https://www.programmersought.com/article/53375975796/)
