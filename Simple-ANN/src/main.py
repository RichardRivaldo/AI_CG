# Libraries
from ann import ArtificialNeuralNetwork
import sys, getopt

# Usage Examples
# python src/main.py --dataset data\heart.csv
# python src/main.py --dataset data\heart.csv --lr 0.9
# python src/main.py --dataset data\heart.csv --batch_size 2 --epochs 2000


if __name__ == "__main__":
    # Initialize getopt parameters
    arg_list = sys.argv[1:]
    long_opt = ["dataset=", "lr=", "epochs=", "batch_size="]

    # Get all arguments
    try:
        args, values = getopt.getopt(args=arg_list, shortopts="", longopts=long_opt)
    except getopt.error as err:
        print(str(err))

    # No args handler
    if len(arg_list) < 1:
        print("Invalid Call Arguments!")
        sys.exit()

    # Initialize parameters for all algorithms with default value
    dataset = None
    lr = 0.1
    epochs = 1000
    batch_size = 1

    # Iterate over the arguments list
    for currArgs, currVal in args:
        # Get all given arguments
        # Only dataset will be the compulsory argumentsss
        currArgs = str(currArgs).strip()
        if currArgs == "--dataset":
            dataset = currVal
        elif currArgs == "--lr":
            lr = float(currVal)
        elif currArgs == "--epochs":
            epochs = int(currVal)
        elif currArgs == "--batch_size":
            batch_size = int(currVal)

    # Check if the dataset is given
    if not dataset:
        print("No dataset given to the Neural Network!")
        sys.exit()

    # Create the Neural Network object
    ann = ArtificialNeuralNetwork(dataset)

    # Automatically fit and train the Neural Network to the dataset
    ann.fit_train(lr, epochs, batch_size)

    # Make a prediction of random input data
    ann.output_predict()
