from utils.utils import gpu_checking
import os
import pickle
from Model import AE, DAGMM, Boosting_aug, OmniAnomaly, USAD, TadGAN, LSTMAE, LSTMVAE, AE_decom, LSTM_decom, Boosting
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
                self.model['encoder'].load_state_dict(torch.load(
                    f"{self.save_path}{self.args.model}_encoder.pk"))
                self.model['decoder'].load_state_dict(torch.load(
                    f"{self.save_path}{self.args.model}_decoder.pk"))
                self.model['critic_x'].load_state_dict(torch.load(
                    f"{self.save_path}{self.args.model}_critic_x.pk"))
                self.model['critic_z'].load_state_dict(torch.load(
                    f"{self.save_path}{self.args.model}_critic_z.pk"))
            # elif self.args.model == 'AE_decom':
            #     self.model['trend_model'].load_state_dict(torch.load(f"{self.save_path}_trend.pk"))
            #     self.model['seasonal_model'].load_state_dict(torch.load(f"{self.save_path}_seasonal.pk"))
            else:
                self.model.load_state_dict(torch.load(
                    f"{self.save_path}model_{self.args.model}.pk"))

    def __build_model(self, args):
        model = ''

        if self.args.model == 'AE':
            model = AE.AutoEncoder(self.data_info['num_features'],
                                   self.data_info['seq_len']).to(self.device)
        elif self.args.model == 'DAGMM':
            model = DAGMM.DAGMM(self.data_info['num_features'],
                                self.data_info['seq_len']).to(self.device)
        elif self.args.model == 'OmniAnomaly':
            model = OmniAnomaly.OmniAnomaly(
                self.data_info['num_features']).to(self.device)
        elif self.args.model == 'USAD':
            model = USAD.USAD(self.data_info['num_features'],
                              self.data_info['seq_len']).to(self.device)
        elif self.args.model == 'LSTMAE':
            model = LSTMAE.RecurrentAutoencoder(self.data_info['seq_len'],
                                                self.data_info['num_features'],
                                                device=self.device).to(self.device)
        elif self.args.model == 'LSTMVAE':
            model = LSTMVAE.RNNPredictor('LSTM', self.data_info['seq_len'] * self.data_info['num_features'],
                                         50).to(self.device)
        elif self.args.model == 'TadGAN':
            encoder = TadGAN.Encoder(
                self.data_info['num_features']*self.data_info['seq_len']).to(self.device)
            decoder = TadGAN.Decoder(
                self.data_info['num_features']*self.data_info['seq_len']).to(self.device)
            critic_x = TadGAN.CriticX(
                self.data_info['num_features']*self.data_info['seq_len']).to(self.device)
            critic_z = TadGAN.CriticZ(
                self.data_info['num_features']*self.data_info['seq_len']).to(self.device)
            model = {'encoder': encoder,
                     'decoder': decoder,
                     'critic_x': critic_x,
                     'critic_z': critic_z}
        elif self.args.model == 'AE_decom':
            model = AE_decom.Model(
                self.args, self.data_info['num_features'], self.device).to(self.device)
            # trend_model = AE.AutoEncoder(self.data_info['num_features'],
            #                             self.data_info['seq_len']).to(self.device)
            # seasonal_model = AE.AutoEncoder(self.data_info['num_features'],
            #                                 self.data_info['seq_len']).to(self.device)
            # model = {'trend_model':trend_model,
            #         'seasonal_model':seasonal_model}
        elif self.args.model == 'LSTM_decom':
            model = LSTM_decom.Model(
                self.args, self.data_info['num_features'], self.device).to(self.device)
        elif self.args.model == 'Boosting':
            model = Boosting.Model(self.data_info['seq_len'], self.data_info['num_features'],
                                   stack_num=2, device=self.device).to(self.device)
        elif self.args.model == 'Boosting_aug':
            model = Boosting_aug.Model(self.data_info['seq_len'], self.data_info['num_features'],
                                          stack_num=2, device=self.device).to(self.device)
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
