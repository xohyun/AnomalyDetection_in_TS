from utils.utils import gpu_checking
import os
import pickle
from Model import AE, DAGMM, OmniAnomaly, USAD, TadGAN
from utils.utils import create_folder
import torch

class ModelMaker:
    def __init__(self, args, data_info):
        self.args = args
        self.data_info = data_info

        print(f"Model setting {self.args.model} ...")
        
        self.device = gpu_checking(self.args)
        self.save_path = self.args.save_path

        self.model = self.__build_model(self.args)
        if self.args.mode == "test":
            # self.model = pretrained_model(self.args.save_path, self.args.model)
            if self.args.model == "TadGAN":
                self.model['encoder'].load_state_dict(torch.load(f"{self.save_path}{self.args.model}_encoder.pk"))
                self.model['decoder'].load_state_dict(torch.load(f"{self.save_path}{self.args.model}_decoder.pk"))
                self.model['critic_x'].load_state_dict(torch.load(f"{self.save_path}{self.args.model}_critic_x.pk"))
                self.model['critic_z'].load_state_dict(torch.load(f"{self.save_path}{self.args.model}_critic_z.pk"))
            else:
                self.model.load_state_dict(torch.load(f"{self.save_path}model_{self.args.model}.pk"))
        
    def __build_model(self, args):
        model = ''

        if self.args.model == 'AE':
            model = AE.AutoEncoder(self.data_info['num_features'],
                                    self.data_info['seq_len']).to(self.device)
        elif self.args.model == 'DAGMM':
            model = DAGMM.DAGMM(self.data_info['num_features'],
                                self.data_info['seq_len']).to(self.device)
        elif self.args.model == 'OmniAnomaly':
            model = OmniAnomaly.OmniAnomaly(self.data_info['num_features']).to(self.device)
        elif self.args.model == 'USAD':
            model = USAD.USAD(self.data_info['num_features'],
                                self.data_info['seq_len']).to(self.device)
        elif self.args.model == "TadGAN":
            encoder = TadGAN.Encoder(self.data_info['num_features']*self.data_info['seq_len'])
            decoder = TadGAN.Decoder(self.data_info['num_features']*self.data_info['seq_len'])
            critic_x = TadGAN.CriticX(self.data_info['num_features']*self.data_info['seq_len'])
            critic_z = TadGAN.CriticZ(self.data_info['num_features']*self.data_info['seq_len'])
            model = {'encoder':encoder,
                    'decoder':decoder,
                    'critic_x':critic_x,
                    'critic_z':critic_z}
        create_folder(self.save_path)
        # write_pickle(os.path.join(self.save_path, f"model_{self.args.model}.pk"), model)
        return model


def write_pickle(path, data):
    with open(path, 'wb') as f: 
        pickle.dump(data, f)

def read_pickle(path):
    with open(path, "rb") as f:
        data = pickle.load(f)
    return data

def pretrained_model(save_path, model):
    try:
        print("[read save model]")
        model = read_pickle(os.path.join(save_path, f'model_{model}.pk'))
    except FileNotFoundError:
        raise FileNotFoundError

    # model.load_state_dict
    # model = load_model(model, save_path)
    return model

