import argparse

class Args:
    def __init__(self):
        self.args = self.parse_arguments()

    def parse_arguments(self):
        parser = argparse.ArgumentParser()
        
        #---# Mode #---#
        parser.add_argument("--mode", default="train", choices=["train", "test"])
        parser.add_argument("--seed", default=1004, type=int)

        #---# Path #---#
        parser.add_argument("--data_path", default="./TimeSeries-Anomaly-Detection-Dataset/data/")

        #---# Dataset #---#
        parser.add_argument("--dataset", default='NAB', choices=['MSL', 'NAB', 'SMAP', 'SMD', 'WADI'])

        #---# Dataset setting #---#
        parser.add_argument("--seq_len", default=1)
        parser.add_argument("--pred_len", default=1)

        #---# Model #---#
        parser.add_argument("--model", type=str, choices=["aaa"])


        args = parser.parse_args()
        return args

