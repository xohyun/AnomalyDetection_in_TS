from utils import gpu_checking
import os
import pickle

class ModelMaker:
    def __init__(self, args):
        self.args = args
        print(f"Model setting {self.args.model} ...")
        self.device = gpu_checking(self.args)
        self.model = self.__build_model(args)
    
    def __build_model(self, args):
        model = ''

        if self.model == 'AE':
            model = AutoEncoder()
        
        write_pickle(os.path.join(self.args.save_path, f"model_{self.args.standard}.pk"), model)


def write_pickle(path, data):
    with open(path, 'wb') as f:
        pickle.dump(data, f)

def read_pickle(path):
    with open(path, "rb") as f:
        data = pickle.load(f)
    return data

