import torch
import torch.nn as nn
import logging
import torch.optim as optim
from torch.autograd import Variable
from scipy import stats
from utils.utils import gpu_checking
from Trainer.base_trainer import base_trainer
from sklearn.metrics import f1_score, precision_score, recall_score
import numpy as np
import wandb

class TrainMaker(base_trainer):
    def __init__(self, args, model, data_loaders, data_info):
        self.args = args
        self.model = model
        self.mode = self.args.mode
        self.num_featrues = data_info['num_features']
        self.seq_len = data_info['seq_len']
        self.features = self.num_featrues * self.seq_len

        self.batch_size = self.args.batch_size
        self.latent_space_dim = 20 ################# 이건 우선 설정

        self.encoder = self.model['encoder']
        self.decoder = self.model['decoder']
        self.critic_x = self.model['critic_x']
        self.critic_z = self.model['critic_z']

        if self.mode == "train" or self.mode == "all":
            self.data_loader = data_loaders['train']
            if self.args.valid_setting:
                self.data_loader_valid = data_loaders['valid']
        else:
            self.data_loader_test = data_loaders['test']
  
        self.device =  gpu_checking(args)
        self.epoch = self.args.epoch
        
        self.lr = self.args.lr
        self.wd = self.args.wd

        # self.optimizer = getattr(torch.optim, self.args.optimizer)
        # self.optimizer = self.optimizer(self.model.parameters(), lr=self.lr, weight_decay=self.wd)
        
        self.optim_enc = optim.Adam(self.encoder.parameters(), lr=self.lr, weight_decay=self.wd)
        self.optim_dec = optim.Adam(self.decoder.parameters(), lr=self.lr, weight_decay=self.wd)
        self.optim_cx = optim.Adam(self.critic_x.parameters(), lr=self.lr, weight_decay=self.wd)
        self.optim_cz = optim.Adam(self.critic_z.parameters(), lr=self.lr, weight_decay=self.wd)

        self.criterion = self.set_criterion(self.args.criterion)
        # self.scheduler = self.set_scheduler(args, self.optimizer)

    def train(self):
        logging.basicConfig(level=logging.DEBUG)
        logging.debug('Starting training')
        cx_epoch_loss = list()
        cz_epoch_loss = list()
        encoder_epoch_loss = list()
        decoder_epoch_loss = list()

        for epoch in range(self.epoch):
            logging.debug('Epoch {}'.format(epoch))
            n_critics = 5

            cx_nc_loss = list()
            cz_nc_loss = list()

            for i in range(n_critics):
                cx_loss = list()
                cz_loss = list()

                for batch, sample in enumerate(self.data_loader):
                    loss = self.critic_x_iteration(sample.to(device = self.device))
                    cx_loss.append(loss)

                    loss = self.critic_z_iteration(sample.to(device = self.device))
                    cz_loss.append(loss)

                cx_nc_loss.append(torch.mean(torch.tensor(cx_loss)))
                cz_nc_loss.append(torch.mean(torch.tensor(cz_loss)))
                
            # print(f"{epoch}epoch loss : {torch.mean(torch.tensor(cx_nc_loss))}")
            
            logging.debug('Critic training done in epoch {}'.format(epoch))
            encoder_loss = list()
            decoder_loss = list()

            for batch, sample in enumerate(self.data_loader):
                enc_loss = self.encoder_iteration(sample.to(device = self.device))
                dec_loss = self.decoder_iteration(sample.to(device = self.device))
                encoder_loss.append(enc_loss)
                decoder_loss.append(dec_loss)

            wandb.log({"cx_nc_loss":torch.mean(torch.tensor(cx_nc_loss)),
                        "cz_nc_loss":torch.mean(torch.tensor(cz_nc_loss)),
                        "encoder_loss":torch.mean(torch.tensor(encoder_loss)),
                        "decoder_loss":torch.mean(torch.tensor(decoder_loss))})
            cx_epoch_loss.append(torch.mean(torch.tensor(cx_nc_loss)))
            cz_epoch_loss.append(torch.mean(torch.tensor(cz_nc_loss)))
            encoder_epoch_loss.append(torch.mean(torch.tensor(encoder_loss)))
            decoder_epoch_loss.append(torch.mean(torch.tensor(decoder_loss)))
            logging.debug('Encoder decoder training done in epoch {}'.format(epoch))
            logging.debug('critic x loss {:.3f} critic z loss {:.3f} \nencoder loss {:.3f} decoder loss {:.3f}\n'.format(cx_epoch_loss[-1], cz_epoch_loss[-1], encoder_epoch_loss[-1], decoder_epoch_loss[-1]))

            #---# For save #---#
            torch.save(self.encoder.state_dict(), f'{self.args.save_path}{self.args.model}_encoder.pk')
            torch.save(self.decoder.state_dict(), f'{self.args.save_path}{self.args.model}_decoder.pk')
            torch.save(self.critic_x.state_dict(), f'{self.args.save_path}{self.args.model}_critic_x.pk')
            torch.save(self.critic_z.state_dict(), f'{self.args.save_path}{self.args.model}_critic_z.pk')
    
    def evaluation(self, test_loader):
        #---# prediction #---#
        xs = []
        preds = []
        critic_score = []

        true_list = []
        for batch, (x, y) in enumerate(test_loader):
            xs.append(x)
            x_ = x.view(1, x.shape[0], self.features)
            pred = self.decoder(self.encoder(x_))
            critic = self.critic_x(x_).detach().numpy() # [1,64,1]

            preds.append(pred.reshape(-1, self.seq_len, self.num_featrues))
            critic_score.extend(critic.reshape(-1))

            y = y.reshape(y.shape[0], -1)
            y = y.mean(axis=1).numpy()
            y = np.where(y>0, 1, 0)

            true_list.extend(y)

        predictions = torch.cat(preds)
        xs = torch.cat(xs)
        print(predictions.shape)
        
        #---# calculate error range #---#
        final_scores, true_index, true, predictions = score_anomalies(y = xs.cpu().detach().numpy(),
                                                                    y_hat = predictions.cpu().detach().numpy(),
                                                                    critic = critic_score,
                                                                    index=np.array(range(len(predictions))))

        error_range = find_anomalies(final_scores, index=range(len(final_scores)), anomaly_padding=3, window_size_portion=0.33, window_step_size_portion=0.1)        
        pred_list = np.zeros(len(true_list))
        for i in error_range:
            start = int(i[0])
            end = int(i[1])
            pred_list[start:end] = 1

        # drawing(config, anomaly_value, pd.DataFrame(test_dataset.scaled_test))
        f1 = f1_score(true_list, pred_list, average='macro')
        precision = precision_score(true_list, pred_list, average="macro")
        recall = recall_score(true_list, pred_list, average="macro")
        
        return f1, precision, recall

    def critic_x_iteration(self, sample):
        self.optim_cx.zero_grad()
        
        # x = sample['signal'].view(1, batch_size, signal_shape) ## 원래
        # x = sample.view(1, self.batch_size, self.features)
        x = sample.view(1, sample.shape[0], self.features).to(device = self.device)
        valid_x = self.critic_x(x)
        valid_x = torch.squeeze(valid_x)
        critic_score_valid_x = torch.mean(torch.ones(valid_x.shape).to(device = self.device) * valid_x) # Wasserstein Loss

        #The sampled z are the anomalous points - points deviating from actual distribution of z (obtained through encoding x)
        z = torch.empty(1, self.batch_size, self.latent_space_dim).uniform_(0, 1)
        x_ = self.decoder(z.to(device = self.device))
        fake_x = self.critic_x(x_)
        fake_x = torch.squeeze(fake_x)
        critic_score_fake_x = torch.mean(torch.ones(fake_x.shape).to(device = self.device) * fake_x)  #Wasserstein Loss

        alpha = torch.rand(x.shape).to(device = self.device)
        ix = Variable(alpha * x + (1 - alpha) * x_) #Random Weighted Average
        ix.requires_grad_(True)
        v_ix = self.critic_x(ix)
        v_ix.mean().backward()
        gradients = ix.grad
        #Gradient Penalty Loss
        gp_loss = torch.sqrt(torch.sum(torch.square(gradients).view(-1)))

        #Critic has to maximize Cx(Valid X) - Cx(Fake X).
        #Maximizing the above is same as minimizing the negative.
        wl = critic_score_fake_x - critic_score_valid_x
        loss = wl + gp_loss
        loss.backward()
        self.optim_cx.step()

        return loss

    def critic_z_iteration(self, sample):
        self.optim_cz.zero_grad()

        # x = sample['signal'].view(1, batch_size, signal_shape)
        x = sample.view(1, self.batch_size, self.features).to(device = self.device)
        z = self.encoder(x)
        valid_z = self.critic_z(z)
        valid_z = torch.squeeze(valid_z)
        critic_score_valid_z = torch.mean(torch.ones(valid_z.shape).to(device = self.device) * valid_z)

        z_ = torch.empty(1, self.batch_size, self.latent_space_dim).uniform_(0, 1).to(device = self.device)
        fake_z = self.critic_z(z_)
        fake_z = torch.squeeze(fake_z)
        critic_score_fake_z = torch.mean(torch.ones(fake_z.shape).to(device = self.device) * fake_z) #Wasserstein Loss

        wl = critic_score_fake_z - critic_score_valid_z

        alpha = torch.rand(z.shape).to(device = self.device)
        iz = Variable(alpha * z + (1 - alpha) * z_) #Random Weighted Average
        iz.requires_grad_(True)
        v_iz = self.critic_z(iz)
        v_iz.mean().backward()
        gradients = iz.grad
        gp_loss = torch.sqrt(torch.sum(torch.square(gradients).view(-1)))

        loss = wl + gp_loss
        loss.backward()
        self.optim_cz.step()

        return loss

    def encoder_iteration(self, sample):
        self.optim_enc.zero_grad()
        # x = sample['signal'].view(1, batch_size, signal_shape) # 원래
        # x = sample.view(1, self.batch_size, self.features)
        x = sample.view(1, sample.shape[0], self.features).to(device = self.device)
        valid_x = self.critic_x(x)
        valid_x = torch.squeeze(valid_x)
        critic_score_valid_x = torch.mean(torch.ones(valid_x.shape).to(device = self.device) * valid_x) #Wasserstein Loss

        z = torch.empty(1, self.batch_size, self.latent_space_dim).uniform_(0, 1).to(device = self.device)
        x_ = self.decoder(z)
        fake_x = self.critic_x(x_)
        fake_x = torch.squeeze(fake_x)
        critic_score_fake_x = torch.mean(torch.ones(fake_x.shape).to(device = self.device) * fake_x)

        enc_z = self.encoder(x)
        gen_x = self.decoder(enc_z)

        mse = self.criterion(x.float(), gen_x.float())
        loss_enc = mse + critic_score_valid_x - critic_score_fake_x
        loss_enc.backward(retain_graph=True)
        self.optim_enc.step()

        return loss_enc

    def decoder_iteration(self, sample):
        self.optim_dec.zero_grad()

        # x = sample['signal'].view(1, batch_size, signal_shape)
        x = sample.view(1, self.batch_size, self.features).to(device = self.device)
        z = self.encoder(x)
        valid_z = self.critic_z(z)
        valid_z = torch.squeeze(valid_z)
        critic_score_valid_z = torch.mean(torch.ones(valid_z.shape).to(device = self.device) * valid_z)

        z_ = torch.empty(1, self.batch_size, self.latent_space_dim).uniform_(0, 1).to(device = self.device)
        fake_z = self.critic_z(z_)
        fake_z = torch.squeeze(fake_z)
        critic_score_fake_z = torch.mean(torch.ones(fake_z.shape).to(device = self.device) * fake_z)

        enc_z = self.encoder(x)
        gen_x = self.decoder(enc_z)

        mse = self.criterion(x.float(), gen_x.float())
        loss_dec = mse + critic_score_valid_z - critic_score_fake_z
        loss_dec.backward(retain_graph=True)
        self.optim_dec.step()

        return loss_dec

    def set_criterion(self, criterion):
        if criterion == "MSE":
            criterion = nn.MSELoss()
        elif criterion == "CEE":
            criterion = nn.CrossEntropyLoss()
        elif criterion == "cosine":
            criterion = nn.CosineEmbeddingLoss()
            
        return criterion

    def set_scheduler(self, args, optimizer):
        if args.scheduler is None:
            return None
        elif args.scheduler == 'exp':
            scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=args.gamma)
        elif args.scheduler == 'step':
            scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)
        elif args.scheduler == 'multi_step':
            scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.milestones, gamma=args.gamma)
        elif args.scheduler == 'plateau':
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=20,
                                                             threshold=0.1, threshold_mode='abs', verbose=True)
        elif args.scheduler == 'cosine':
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                             T_max=args.T_max if args.T_max else args.epochs,
                                                             eta_min=args.eta_min if args.eta_min else 0)
        elif args.scheduler == 'one_cycle':
            scheduler = optim.lr_scheduler.OneCycleLR(optimizer,
                                                    max_lr=args.max_lr, 
                                                    steps_per_epoch=args.steps_per_epoch,
                                                    epochs=args.cycle_epochs)
        else:
            raise ValueError(f"Not supported {args.scheduler}.")
        return scheduler

"""
Time Series error calculation functions.
"""

import math

import numpy as np
import pandas as pd
# from pyts.metrics import dtw
from scipy import integrate


def regression_errors(y, y_hat, smoothing_window=0.01, smooth=True):
    """Compute an array of absolute errors comparing predictions and expected output.
    If smooth is True, apply EWMA to the resulting array of errors.
    Args:
        y (ndarray):
        Ground truth.
        y_hat (ndarray):
        Predicted values.
        smoothing_window (float):
        Optional. Size of the smoothing window, expressed as a proportion of the total
        length of y. If not given, 0.01 is used.
        smooth (bool):
        Optional. Indicates whether the returned errors should be smoothed with EWMA.
        If not given, `True` is used.
    Returns:
        ndarray:
        Array of errors.
    """
    errors = np.abs(y - y_hat)[:, 0]

    if not smooth:
        return errors

    smoothing_window = int(smoothing_window * len(y))

    return pd.Series(errors).ewm(span=smoothing_window).mean().values


def _point_wise_error(y, y_hat):
    """Compute point-wise error between predicted and expected values.
    The computed error is calculated as the difference between predicted
    and expected values with a rolling smoothing factor.
    Args:
        y (ndarray):
        Ground truth.
        y_hat (ndarray):
        Predicted values.
    Returns:
        ndarray:
        An array of smoothed point-wise error.
    """
    return abs(y - y_hat)


def _area_error(y, y_hat, score_window=10):
    """Compute area error between predicted and expected values.
    The computed error is calculated as the area difference between predicted
    and expected values with a smoothing factor.
    Args:
        y (ndarray):
        Ground truth.
        y_hat (ndarray):
        Predicted values.
        score_window (int):
        Optional. Size of the window over which the scores are calculated.
        If not given, 10 is used.
    Returns:
        ndarray:
        An array of area error.
    """
    smooth_y = pd.Series(y).rolling(
        score_window, center=True, min_periods=score_window // 2).apply(integrate.trapz)
    smooth_y_hat = pd.Series(y_hat).rolling(
        score_window, center=True, min_periods=score_window // 2).apply(integrate.trapz)

    errors = abs(smooth_y - smooth_y_hat)

    return errors


def _dtw_error(y, y_hat, score_window=10):
    """Compute dtw error between predicted and expected values.
    The computed error is calculated as the dynamic time warping distance
    between predicted and expected values with a smoothing factor.
    Args:
        y (ndarray):
        Ground truth.
        y_hat (ndarray):
        Predicted values.
        score_window (int):
        Optional. Size of the window over which the scores are calculated.
        If not given, 10 is used.
    Returns:
        ndarray:
        An array of dtw error.
    """
    length_dtw = (score_window // 2) * 2 + 1
    half_length_dtw = length_dtw // 2

    # add padding
    y_pad = np.pad(y, (half_length_dtw, half_length_dtw),
                    'constant', constant_values=(0, 0))
    y_hat_pad = np.pad(y_hat, (half_length_dtw, half_length_dtw),
                    'constant', constant_values=(0, 0))

    i = 0
    similarity_dtw = list()
    while i < len(y) - length_dtw:
        true_data = y_pad[i:i + length_dtw]
        true_data = true_data.flatten()

        pred_data = y_hat_pad[i:i + length_dtw]
        pred_data = pred_data.flatten()

        dist = dtw(true_data, pred_data)
        similarity_dtw.append(dist)
        i += 1

    errors = ([0] * half_length_dtw + similarity_dtw +
                [0] * (len(y) - len(similarity_dtw) - half_length_dtw))

    return errors


def reconstruction_errors(y, y_hat, step_size=1, score_window=10, smoothing_window=0.01,
                        smooth=True, rec_error_type='point'):
    """Compute an array of reconstruction errors.
    Compute the discrepancies between the expected and the
    predicted values according to the reconstruction error type.
    Args:
        y (ndarray):
        Ground truth.
        y_hat (ndarray):
        Predicted values. Each timestamp has multiple predictions.
        step_size (int):
        Optional. Indicating the number of steps between windows in the predicted values.
        If not given, 1 is used.
        score_window (int):
        Optional. Size of the window over which the scores are calculated.
        If not given, 10 is used.
        smoothing_window (float or int):
        Optional. Size of the smoothing window, when float it is expressed as a proportion
        of the total length of y. If not given, 0.01 is used.
        smooth (bool):
        Optional. Indicates whether the returned errors should be smoothed.
        If not given, `True` is used.
        rec_error_type (str):
        Optional. Reconstruction error types ``["point", "area", "dtw"]``.
        If not given, "point" is used.
    Returns:
        ndarray:
        Array of reconstruction errors.
    """
    y = y.reshape(y.shape[0], -1, 1) # 추가 부분 y.shape = (932, 1000, 1)
    y_hat = y_hat.reshape(y_hat.shape[0], -1, 1) # 추가 부분 y_hat shape = (932, 1000, 1)
    
    if isinstance(smoothing_window, float):
        smoothing_window = min(math.trunc(len(y) * smoothing_window), 200) # 9
        # len(y) * smoothing_window = 9 (932 * 0.01)

    true = [item[0] for item in y.reshape((y.shape[0], -1))] # 932개

    for item in y[-1][1:]: # y[-1][1] shape : (999,1)
        true.extend(item) # 총 개수 1931
    
    predictions = []
    predictions_vs = []

    pred_length = y_hat.shape[1] # 1000 
    num_errors = y_hat.shape[1] + step_size * (y_hat.shape[0] - 1) # 1931

    for i in range(num_errors): 
        intermediate = []
        for j in range(max(0, i - num_errors + pred_length), min(i + 1, pred_length)):
            intermediate.append(y_hat[i - j, j])

        if intermediate:
            predictions.append(np.median(np.asarray(intermediate)))

            predictions_vs.append([[
                np.min(np.asarray(intermediate)),
                np.percentile(np.asarray(intermediate), 25),
                np.percentile(np.asarray(intermediate), 50),
                np.percentile(np.asarray(intermediate), 75),
                np.max(np.asarray(intermediate))
            ]])

    true = np.asarray(true)
    predictions = np.asarray(predictions)
    predictions_vs = np.asarray(predictions_vs)

    # Compute reconstruction errors
    if rec_error_type.lower() == "point":
        errors = _point_wise_error(true, predictions)

    elif rec_error_type.lower() == "area":
        errors = _area_error(true, predictions, score_window)

    elif rec_error_type.lower() == "dtw":
        errors = _dtw_error(true, predictions, score_window)

    # Apply smoothing
    if smooth:
        errors = pd.Series(errors).rolling(
        smoothing_window, center=True, min_periods=smoothing_window // 2).mean().values

    return errors, predictions_vs

###########

def _compute_critic_score(critics, smooth_window):
    """Compute an array of anomaly scores.
    Args:
        critics (ndarray):
            Critic values.
        smooth_window (int):
            Smooth window that will be applied to compute smooth errors.
    Returns:
        ndarray:
            Array of anomaly scores.
    """
    critics = np.asarray(critics)
    l_quantile = np.quantile(critics, 0.25)
    u_quantile = np.quantile(critics, 0.75)
    in_range = np.logical_and(critics >= l_quantile, critics <= u_quantile)
    critic_mean = np.mean(critics[in_range])
    critic_std = np.std(critics)

    z_scores = np.absolute((np.asarray(critics) - critic_mean) / critic_std) + 1
    z_scores = pd.Series(z_scores).rolling(
        smooth_window, center=True, min_periods=smooth_window // 2).mean().values

    return z_scores



def score_anomalies(y, y_hat, critic, index,
                  score_window: int = 10, critic_smooth_window: int = None,
                  error_smooth_window: int = None, smooth: bool = True,
                  rec_error_type: str = "point", comb: str = "mult",
                  lambda_rec: float = 0.5):
    """Compute an array of anomaly scores.
    Anomaly scores are calculated using a combination of reconstruction error and critic score.
    Args:
        y (ndarray):
            Ground truth.
        y_hat (ndarray):
            Predicted values. Each timestamp has multiple predictions.
        index (ndarray):
            time index for each y (start position of the window)
        critic (ndarray):
            Critic score. Each timestamp has multiple critic scores.
        score_window (int):
            Optional. Size of the window over which the scores are calculated.
            If not given, 10 is used.
        critic_smooth_window (int):
            Optional. Size of window over which smoothing is applied to critic.
            If not given, 200 is used.
        error_smooth_window (int):
            Optional. Size of window over which smoothing is applied to error.
            If not given, 200 is used.
        smooth (bool):
            Optional. Indicates whether errors should be smoothed.
            If not given, `True` is used.
        rec_error_type (str):
            Optional. The method to compute reconstruction error. Can be one of
            `["point", "area", "dtw"]`. If not given, 'point' is used.
        comb (str):
            Optional. How to combine critic and reconstruction error. Can be one
            of `["mult", "sum", "rec"]`. If not given, 'mult' is used.
        lambda_rec (float):
            Optional. Used if `comb="sum"` as a lambda weighted sum to combine
            scores. If not given, 0.5 is used.
    Returns:
        ndarray:
            Array of anomaly scores.
    """
    y = y.reshape(y.shape[0], -1, 1) # 추가 부분 y.shape = (932, 1000, 1)
    y_hat = y_hat.reshape(y_hat.shape[0], -1, 1) # 추가 부분 y_hat shape = (932, 1000, 1)

    critic_smooth_window = critic_smooth_window or math.trunc(y.shape[0] * 0.01)
    error_smooth_window = error_smooth_window or math.trunc(y.shape[0] * 0.01)

    step_size = 1  # expected to be 1

    true_index = index  # no offset

    true = [item[0] for item in y.reshape((y.shape[0], -1))]
    for item in y[-1][1:]:
        true.extend(item)


    critic_extended = list()
    for c in critic:
        critic_extended.extend(np.repeat(c, y_hat.shape[1]).tolist())

    critic_extended = np.asarray(critic_extended).reshape((-1, y_hat.shape[1]))

    critic_kde_max = []
    pred_length = y_hat.shape[1]
    num_errors = y_hat.shape[1] + step_size * (y_hat.shape[0] - 1)

    for i in range(num_errors):
        critic_intermediate = []

        for j in range(max(0, i - num_errors + pred_length), min(i + 1, pred_length)):
            critic_intermediate.append(critic_extended[i - j, j])

        if len(critic_intermediate) > 1:
            discr_intermediate = np.asarray(critic_intermediate)
            try:
                critic_kde_max.append(discr_intermediate[np.argmax(
                    stats.gaussian_kde(discr_intermediate)(critic_intermediate))])
            except np.linalg.LinAlgError:
                critic_kde_max.append(np.median(discr_intermediate))
        else:
            critic_kde_max.append(np.median(np.asarray(critic_intermediate)))

    # Compute critic scores
    critic_scores = _compute_critic_score(critic_kde_max, critic_smooth_window)

    # Compute reconstruction scores
    rec_scores, predictions = reconstruction_errors(
        y, y_hat, step_size, score_window, error_smooth_window, smooth, rec_error_type)

    rec_scores = stats.zscore(rec_scores)
    rec_scores = np.clip(rec_scores, a_min=0, a_max=None) + 1

    # Combine the two scores
    if comb == "mult":
        final_scores = np.multiply(critic_scores, rec_scores)

    elif comb == "sum":
        final_scores = (1 - lambda_rec) * (critic_scores - 1) + lambda_rec * (rec_scores - 1)

    elif comb == "rec":
        final_scores = rec_scores

    else:
        raise ValueError(
        'Unknown combination specified {}, use "mult", "sum", or "rec" instead.'.format(comb))

    true = [[t] for t in true]
    return final_scores, true_index, true, predictions


"""
Time Series anomaly detection functions.
Some of the implementation is inspired by the paper https://arxiv.org/pdf/1802.04431.pdf
"""

import numpy as np
import pandas as pd
from scipy.optimize import fmin


def deltas(errors, epsilon, mean, std):
    """Compute mean and std deltas.
    delta_mean = mean(errors) - mean(all errors below epsilon)
    delta_std = std(errors) - std(all errors below epsilon)
    Args:
        errors (ndarray):
        Array of errors.
        epsilon (ndarray):
        Threshold value.
        mean (float):
        Mean of errors.
        std (float):
        Standard deviation of errors.
    Returns:
        float, float:
        * delta_mean.
        * delta_std.
    """
    below = errors[errors <= epsilon]
    if not len(below):
        return 0, 0

    return mean - below.mean(), std - below.std()


def count_above(errors, epsilon):
    """Count number of errors and continuous sequences above epsilon.
    Continuous sequences are counted by shifting and counting the number
    of positions where there was a change and the original value was true,
    which means that a sequence started at that position.
    Args:
        errors (ndarray):
        Array of errors.
        epsilon (ndarray):
        Threshold value.
    Returns:
        int, int:
        * Number of errors above epsilon.
        * Number of continuous sequences above epsilon.
    """
    above = errors > epsilon
    total_above = len(errors[above])

    above = pd.Series(above)
    shift = above.shift(1)
    change = above != shift

    total_consecutive = sum(above & change)

    return total_above, total_consecutive


def z_cost(z, errors, mean, std):
    """Compute how bad a z value is.
    The original formula is::
                    (delta_mean/mean) + (delta_std/std)
        ------------------------------------------------------
        number of errors above + (number of sequences above)^2
    which computes the "goodness" of `z`, meaning that the higher the value
    the better the `z`.
    In this case, we return this value inverted (we make it negative), to convert
    it into a cost function, as later on we will use scipy.fmin to minimize it.
    Args:
        z (ndarray):
        Value for which a cost score is calculated.
        errors (ndarray):
        Array of errors.
        mean (float):
        Mean of errors.
        std (float):
        Standard deviation of errors.
    Returns:
        float:
        Cost of z.
    """
    epsilon = mean + z * std

    delta_mean, delta_std = deltas(errors, epsilon, mean, std)
    above, consecutive = count_above(errors, epsilon)

    numerator = -(delta_mean / mean + delta_std / std)
    denominator = above + consecutive ** 2

    if denominator == 0:
        return np.inf

    return numerator / denominator


def _find_threshold(errors, z_range):
    """Find the ideal threshold.
    The ideal threshold is the one that minimizes the z_cost function. Scipy.fmin is used
    to find the minimum, using the values from z_range as starting points.
    Args:
        errors (ndarray):
        Array of errors.
        z_range (list):
        List of two values denoting the range out of which the start points for the
        scipy.fmin function are chosen.
    Returns:
        float:
        Calculated threshold value.
    """
    mean = errors.mean()
    std = errors.std()

    min_z, max_z = z_range
    best_z = min_z
    best_cost = np.inf
    for z in range(min_z, max_z):
        best = fmin(z_cost, z, args=(errors, mean, std), full_output=True, disp=False) # minimize
        z, cost = best[0:2]
    
        if cost < best_cost:
            best_z = z[0]

    return mean + best_z * std


def _fixed_threshold(errors, k=4):
    """Calculate the threshold.
    The fixed threshold is defined as k standard deviations away from the mean.
    Args:
        errors (ndarray):
        Array of errors.
    Returns:
        float:
        Calculated threshold value.
    """
    mean = errors.mean()
    std = errors.std()

    return mean + k * std


def _find_sequences(errors, epsilon, anomaly_padding):
    """Find sequences of values that are above epsilon.
    This is done following this steps:
        * create a boolean mask that indicates which values are above epsilon.
        * mark certain range of errors around True values with a True as well.
        * shift this mask by one place, filing the empty gap with a False.
        * compare the shifted mask with the original one to see if there are changes.
        * Consider a sequence start any point which was true and has changed.
        * Consider a sequence end any point which was false and has changed.
    Args:
        errors (ndarray):
        Array of errors.
        epsilon (float):
        Threshold value. All errors above epsilon are considered an anomaly.
        anomaly_padding (int):
        Number of errors before and after a found anomaly that are added to the
        anomalous sequence.
    Returns:
        ndarray, float:
        * Array containing start, end of each found anomalous sequence.
        * Maximum error value that was not considered an anomaly.
    """
    above = pd.Series(errors > epsilon)
    index_above = np.argwhere(above.values)

    for idx in index_above.flatten():
        above[max(0, idx - anomaly_padding):min(idx + anomaly_padding + 1, len(above))] = True

    shift = above.shift(1).fillna(False)
    change = above != shift

    if above.all():
        max_below = 0
    else:
        max_below = max(errors[~above])

    index = above.index
    starts = index[above & change].tolist()
    ends = (index[~above & change] - 1).tolist()

    if len(ends) == len(starts) - 1:
        ends.append(len(above) - 1)

    return np.array([starts, ends]).T, max_below


def _get_max_errors(errors, sequences, max_below):
    """Get the maximum error for each anomalous sequence.
    Also add a row with the max error which was not considered anomalous.
    Table containing a ``max_error`` column with the maximum error of each
    sequence and the columns ``start`` and ``stop`` with the corresponding start and stop
    indexes, sorted descendingly by the maximum error.
    Args:
        errors (ndarray):
        Array of errors.
        sequences (ndarray):
        Array containing start, end of anomalous sequences
        max_below (float):
        Maximum error value that was not considered an anomaly.
    Returns:
        pandas.DataFrame:
        DataFrame object containing columns ``start``, ``stop`` and ``max_error``.
    """
    max_errors = [{
        'max_error': max_below,
        'start': -1,
        'stop': -1
    }]

    for sequence in sequences:
        start, stop = sequence
        sequence_errors = errors[start: stop + 1]
        max_errors.append({
            'start': start,
            'stop': stop,
            'max_error': max(sequence_errors)
        })

    max_errors = pd.DataFrame(max_errors).sort_values('max_error', ascending=False)
    return max_errors.reset_index(drop=True)


def _prune_anomalies(max_errors, min_percent):
    """Prune anomalies to mitigate false positives.
    This is done by following these steps:
        * Shift the errors 1 negative step to compare each value with the next one.
        * Drop the last row, which we do not want to compare.
        * Calculate the percentage increase for each row.
        * Find rows which are below ``min_percent``.
        * Find the index of the latest of such rows.
        * Get the values of all the sequences above that index.
    Args:
        max_errors (pandas.DataFrame):
        DataFrame object containing columns ``start``, ``stop`` and ``max_error``.
        min_percent (float):
        Percentage of separation the anomalies need to meet between themselves and the
        highest non-anomalous error in the window sequence.
    Returns:
        ndarray:
        Array containing start, end, max_error of the pruned anomalies.
    """
    next_error = max_errors['max_error'].shift(-1).iloc[:-1]
    max_error = max_errors['max_error'].iloc[:-1]

    increase = (max_error - next_error) / max_error
    too_small = increase < min_percent

    if too_small.all():
        last_index = -1
    else:
        last_index = max_error[~too_small].index[-1]

    return max_errors[['start', 'stop', 'max_error']].iloc[0: last_index + 1].values


def _compute_scores(pruned_anomalies, errors, threshold, window_start):
    """Compute the score of the anomalies.
    Calculate the score of the anomalies proportional to the maximum error in the sequence
    and add window_start timestamp to make the index absolute.
    Args:
        pruned_anomalies (ndarray):
        Array of anomalies containing the start, end and max_error for all anomalies in
            the window.
        errors (ndarray):
            Array of errors.
        threshold (float):
            Threshold value.
        window_start (int):
            Index of the first error value in the window.
    Returns:
        list:
        List of anomalies containing start-index, end-index, score for each anomaly.
    """
    anomalies = list()
    denominator = errors.mean() + errors.std()

    for row in pruned_anomalies:
        max_error = row[2]
        score = (max_error - threshold) / denominator
        anomalies.append([row[0] + window_start, row[1] + window_start, score])

    return anomalies


def _merge_sequences(sequences):
    """Merge consecutive and overlapping sequences.
    We iterate over a list of start, end, score triples and merge together
    overlapping or consecutive sequences.
    The score of a merged sequence is the average of the single scores,
    weighted by the length of the corresponding sequences.
    Args:
        sequences (list):
        List of anomalies, containing start-index, end-index, score for each anomaly.
    Returns:
        ndarray:
        Array containing start-index, end-index, score for each anomaly after merging.
    """
    if len(sequences) == 0:
        return np.array([])

    sorted_sequences = sorted(sequences, key=lambda entry: entry[0])
    new_sequences = [sorted_sequences[0]]
    score = [sorted_sequences[0][2]]
    weights = [sorted_sequences[0][1] - sorted_sequences[0][0]]

    for sequence in sorted_sequences[1:]:
        prev_sequence = new_sequences[-1]

        if sequence[0] <= prev_sequence[1] + 1:
            score.append(sequence[2])
            weights.append(sequence[1] - sequence[0])
            weighted_average = np.average(score, weights=weights)
            new_sequences[-1] = (prev_sequence[0], max(prev_sequence[1], sequence[1]),
                                    weighted_average)
        else:
            score = [sequence[2]]
            weights = [sequence[1] - sequence[0]]
            new_sequences.append(sequence)

    return np.array(new_sequences)


def _find_window_sequences(window, z_range, anomaly_padding, min_percent, window_start,
                           fixed_threshold):
    """Find sequences of values that are anomalous.
    We first find the threshold for the window, then find all sequences above that threshold.
    After that, we get the max errors of the sequences and prune the anomalies. Lastly, the
    score of the anomalies is computed.
    Args:
        window (ndarray):
        Array of errors in the window that is analyzed.
        z_range (list):
        List of two values denoting the range out of which the start points for the
        dynamic find_threshold function are chosen.
        anomaly_padding (int):
        Number of errors before and after a found anomaly that are added to the anomalous
        sequence.
        min_percent (float):
        Percentage of separation the anomalies need to meet between themselves and the
        highest non-anomalous error in the window sequence.
        window_start (int):
        Index of the first error value in the window.
        fixed_threshold (bool):
        Indicates whether to use fixed threshold or dynamic threshold.
    Returns:
        ndarray:
        Array containing the start-index, end-index, score for each anomalous sequence
        that was found in the window.
    """
    if fixed_threshold:
        threshold = _fixed_threshold(window)

    else:
        threshold = _find_threshold(window, z_range)

    window_sequences, max_below = _find_sequences(window, threshold, anomaly_padding)
    max_errors = _get_max_errors(window, window_sequences, max_below)
    pruned_anomalies = _prune_anomalies(max_errors, min_percent)
    window_sequences = _compute_scores(pruned_anomalies, window, threshold, window_start)

    return window_sequences


def find_anomalies(errors, index, z_range=(0, 10), window_size=None, window_size_portion=None,
                   window_step_size=None, window_step_size_portion=None, min_percent=0.1,
                   anomaly_padding=50, lower_threshold=False, fixed_threshold=None):
    """Find sequences of error values that are anomalous.
    We first define the window of errors, that we want to analyze. We then find the anomalous
    sequences in that window and store the start/stop index pairs that correspond to each
    sequence, along with its score. Optionally, we can flip the error sequence around the mean
    and apply the same procedure, allowing us to find unusually low error sequences.
    We then move the window and repeat the procedure.
    Lastly, we combine overlapping or consecutive sequences.
    Args:
        errors (ndarray):
        Array of errors.
        index (ndarray):
        Array of indices of the errors.
        z_range (list):
        Optional. List of two values denoting the range out of which the start points for
        the scipy.fmin function are chosen. If not given, (0, 10) is used.
        window_size (int):
        Optional. Size of the window for which a threshold is calculated. If not given,
        `None` is used, which finds one threshold for the entire sequence of errors.
        window_size_portion (float):
        Optional. Specify the size of the window to be a portion of the sequence of errors.
        If not given, `None` is used, and window size is used as is.
        window_step_size (int):
        Optional. Number of steps the window is moved before another threshold is
        calculated for the new window.
        window_step_size_portion (float):
        Optional. Specify the number of steps to be a portion of the window size. If not given,
        `None` is used, and window step size is used as is.
        min_percent (float):
        Optional. Percentage of separation the anomalies need to meet between themselves and
        the highest non-anomalous error in the window sequence. It nof given, 0.1 is used.
        anomaly_padding (int):
        Optional. Number of errors before and after a found anomaly that are added to the
        anomalous sequence. If not given, 50 is used.
        lower_threshold (bool):
        Optional. Indicates whether to apply a lower threshold to find unusually low errors.
        If not given, `False` is used.
        fixed_threshold (bool):
        Optional. Indicates whether to use fixed threshold or dynamic threshold. If not
        given, `False` is used.
    Returns:
        ndarray:
        Array containing start-index, end-index, score for each anomalous sequence that
        was found.
    """
    window_size = window_size or len(errors)
    if window_size_portion:
        window_size = np.ceil(len(errors) * window_size_portion).astype('int')

    window_step_size = window_step_size or window_size
    if window_step_size_portion:
        window_step_size = np.ceil(window_size * window_step_size_portion).astype('int')

    window_start = 0
    window_end = 0
    sequences = list()

    while window_end < len(errors):
        window_end = window_start + window_size
        window = errors[window_start:window_end]
        window_sequences = _find_window_sequences(window, z_range, anomaly_padding, min_percent,
                                                window_start, fixed_threshold)
        sequences.extend(window_sequences)

        if lower_threshold:
            # Flip errors sequence around mean
            mean = window.mean()
            inverted_window = mean - (window - mean)
            inverted_window_sequences = _find_window_sequences(inverted_window, z_range,
                                                                anomaly_padding, min_percent,
                                                                window_start, fixed_threshold)
            sequences.extend(inverted_window_sequences)

        window_start = window_start + window_step_size

    sequences = _merge_sequences(sequences)

    anomalies = list()

    for start, stop, score in sequences:
        anomalies.append([index[int(start)], index[int(stop)], score])

    return np.asarray(anomalies)