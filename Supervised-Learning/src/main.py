# Libraries
from knn import *
from id3 import *
import sys, getopt
from os import name
from logistic_regression import *

# Usage Example:
# python src\main.py --knn --data data\dataset.csv --k_nearest 3
# python src\main.py --log_reg --data data\dataset.csv --lr 0.01 --epochs 1000
# python src\main.py --id_3 --data data\dataset.csv

if __name__ == "__main__":
    # Initialize getopt parameters
    arg_list = sys.argv[1:]
    long_opt = [
        "knn",
        "log_reg",
        "id_3",
        "data =",
        "k_nearest =",
        "lr =",
        "epochs =",
    ]

    # Get all arguments
    try:
        args, values = getopt.getopt(args=arg_list, shortopts="", longopts=long_opt)
    except getopt.error as err:
        print(str(err))

    # No args handler
    if len(arg_list) < 1:
        print("Invalid Call Arguments!")
        sys.exit()

    # Initialize parameters for all algorithms
    algorithm = None
    dataset = None
    k_nearest = None
    lr = None
    epochs = None

    # ALGORITHM TYPE SHOULD BE GIVEN FIRST
    # Determine which algorithm the user want to use
    if args[0][0].strip() in ["--knn", "--log_reg", "--id_3"]:
        algorithm = str(args[0][0]).strip()

    # Get common arguments (data and label)
    for currArgs, currVal in args[1:]:
        currArgs = str(currArgs).strip()
        if currArgs == "--data":
            dataset = currVal

    # Check algorithm
    if algorithm != None:
        # Random input data index
        data_idx = np.random.randint(0, 100)

        # KNN Algorithm
        if algorithm == "--knn":
            # Get and set all needed arguments
            for currArgs, currVal in args[1:]:
                currArgs = str(currArgs).strip()
                if currArgs == "--k_nearest":
                    k_nearest = int(currVal)
            knn_params = [dataset, k_nearest]

            # Check if any argument is not given
            if None not in knn_params:
                # All arguments given, make a random prediction
                knn = KNNClassifier(dataset, k_nearest)
                knn.output_prediction(knn.test_dataset[data_idx])
            else:
                # Not all arguments given
                print("Not all arguments are given (--data, --k_nearest)!")
        # Logistic Regression Algorithm
        elif algorithm == "--log_reg":
            for currArgs, currVal in args[1:]:
                currArgs = str(currArgs).strip()
                if currArgs == "--lr":
                    lr = float(currVal)
                elif currArgs == "--epochs":
                    epochs = int(currVal)
            log_reg_params = [dataset, lr, epochs]

            # Check if any argument is not given
            if None not in log_reg_params:
                # All arguments given, make a random prediction
                logreg = LogisticRegression(dataset, lr, epochs)
                logreg.output_prediction(data_idx)
            else:
                # Not all arguments given
                print("Not all arguments are given (--data, --lr, --epochs)!")
        # ID3 Algorithm
        elif algorithm == "--id_3":
            id3_param = [dataset]

            # Check if any argument is not given
            if None not in id3_param:
                # All arguments given, make a random prediction
                id3_tree = IterativeDichotomiser3(dataset)
                id3_tree.output_prediction(data_idx)
            else:
                # Not all arguments given
                print("Not all arguments are given (--data)!")
    # No algorithm argument provided, or not the first argument in the inputs
    else:
        print("Provide the correct algorithm first (--knn, --log_reg, or --id_3)!")
