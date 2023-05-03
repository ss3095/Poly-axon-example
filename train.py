import pandas as pd
import sys
from surprise import Dataset, Reader, SVDpp, accuracy
from surprise.model_selection import train_test_split

colnames = ['time', 'userid', 'movieid', 'ratings']
df = pd.read_csv('rating_data.csv', names=colnames, header=None, skiprows=1)


def train_eval():
    # A reader is still needed but only the rating_scale param is requiered.
    reader = Reader(rating_scale=(1, 5))

    # The columns must correspond to user id, item id and ratings (in that order).
    dataset = Dataset.load_from_df(df[["userid", "movieid", "ratings"]], reader)

    trainset, testset = train_test_split(dataset, test_size=0.25)
    algo = SVDpp()

    # Train the algorithm on the trainset, and predict ratings for the testset
    algo.fit(trainset)
    predictions = algo.test(testset)

    # Then compute RMSE
    rmse = accuracy.rmse(predictions)

    return rmse


with open("metrics.txt", "w") as outfile:
    outfile.write("RMSE score of the model: " + str(train_eval()) + "\n")
