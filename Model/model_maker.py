from utils import gpu_checking
import os
import pickle
from Model import AE

class ModelMaker:
    def __init__(self, args, data_info, pretrained=False):
        self.args = args
        print(f"Model setting {self.args.model} ...")
        self.device = gpu_checking(self.args)
        self.num_features = data_info
        if pretrained:
            self.model = pretrained_model(self.args.save_path)
        else:
            self.model = self.__build_model(self.args)
        
    def __build_model(self, args):
        model = ''

        if self.args.model == 'AE':
            model = AE.AutoEncoder(self.num_features).to(self.device)

        write_pickle(os.path.join(self.args.save_path, f"model_{1}.pk"), model)
        return model

def write_pickle(path, data):
    with open(path, 'wb') as f:
        pickle.dump(data, f)

def read_pickle(path):
    with open(path, "rb") as f:
        data = pickle.load(f)
    return data

def pretrained_model(save_path):
    try:
        print("[read save model]")
        model = read_pickle(os.path.join(save_path, f'model_{1}.pk'))
    except FileNotFoundError:
        raise FileNotFoundError

    # model.load_state_dict
    # model = load_model(model, save_path)
    return model

