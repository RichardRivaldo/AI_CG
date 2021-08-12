# Libraries
import sys
import getopt
from kmeans import KMeans
from dbscan import DBScan
from kmedoids import KMedoidsPAM

# Usage Example:
# python src\main.py --dbscan --data data\dataset.csv --eps 10 --min_pts 3
# python src\main.py --kmeans --data data\dataset.csv --k_cen 5
# python src\main.py --kmedoids --data data\dataset.csv --k_med 3

if __name__ == "__main__":
    # Initialize getopt parameters
    arg_list = sys.argv[1:]
    long_opt = [
        "dbscan",
        "kmeans",
        "kmedoids",
        "data =",
        "eps =",
        "min_pts =",
        "k_cen =",
        "k_med =",
    ]

    # Get all arguments
    try:
        args, values = getopt.getopt(
            args=arg_list, shortopts="", longopts=long_opt)
    except getopt.error as err:
        print(str(err))

    # No args handler
    if len(arg_list) < 1:
        print("Invalid Call Arguments!")
        sys.exit()

    # Initialize parameters for all algorithms
    algorithm = None
    dataset = None
    epsilon = None
    min_pts = None
    k = None

    # ALGORITHM TYPE SHOULD BE GIVEN FIRST
    # Determine which algorithm the user want to use
    if args[0][0].strip() in ["--dbscan", "--kmeans", "--kmedoids"]:
        algorithm = str(args[0][0]).strip()

    # Get common arguments (data and label)
    for currArgs, currVal in args[1:]:
        currArgs = str(currArgs).strip()
        if currArgs == "--data":
            dataset = currVal

    # Check algorithm
    if algorithm != None:
        # DBScan Algorithm
        if algorithm == "--dbscan":
            # Get and set all needed arguments
            for currArgs, currVal in args[1:]:
                currArgs = str(currArgs).strip()
                if currArgs == "--eps":
                    epsilon = float(currVal)
                elif currArgs == "--min_pts":
                    min_pts = int(currVal)
            dbscan_params = [dataset, epsilon, min_pts]

            # Check if any argument is not given
            if None not in dbscan_params:
                # All arguments given, make a random prediction
                dbscan = DBScan(dataset, epsilon, min_pts)
                dbscan.predict_cluster()
            else:
                # Not all arguments given
                print("Not all arguments are given (--data, --eps, --min_pts)!")
        # KMeans Algorithm
        elif algorithm == "--kmeans":
            for currArgs, currVal in args[1:]:
                currArgs = str(currArgs).strip()
                if currArgs == "--k_cen":
                    k = int(currVal)
            kmeans_params = [dataset, k]

            # Check if any argument is not given
            if None not in kmeans_params:
                # All arguments given, make a random prediction
                kmeans = KMeans(dataset, k)
                kmeans.predict_cluster()
            else:
                # Not all arguments given
                print("Not all arguments are given (--data, --k_cen)!")
        # KMedoids Algorithm
        elif algorithm == "--kmedoids":
            # Get and set all needed arguments
            for currArgs, currVal in args[1:]:
                currArgs = str(currArgs).strip()
                if currArgs == "--k_med":
                    k = int(currVal)
            kmedoids_params = [dataset, k]

            # Check if any argument is not given
            if None not in kmedoids_params:
                # All arguments given, make a random prediction
                kmedoids = KMedoidsPAM(dataset, k)
                kmedoids.predict_cluster()
            else:
                # Not all arguments given
                print("Not all arguments are given (--data, --k_med)!")
    # No algorithm argument provided, or not the first argument in the inputs
    else:
        print(
            "Provide the correct algorithm first (--dbscan, --kmeans, or --kmedoids)!"
        )
