import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import tqdm
import wandb
from tqdm import tqdm
from sklearn.metrics import f1_score, precision_score, recall_score
from utils.utils import gpu_checking, find_bundle
from Trainer.base_trainer import base_trainer
from sklearn.metrics import mean_absolute_error, mean_squared_error

import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

class TrainMaker(base_trainer):
    def __init__(self, args, model, data_loaders, data_info):
        self.args = args
        self.model = model
        self.mode = self.args.mode

        if self.mode == "train" or self.mode == "all":
            self.data_loader = data_loaders['train']
            if self.args.valid_setting:
                self.data_loader_valid = data_loaders['valid']
        else:
            self.data_loader_test = data_loaders['test']
  
        self.device =  gpu_checking(args)

        self.features = data_info['num_features']
        self.epoch = self.args.epoch
        
        self.lr = self.args.lr
        self.wd = self.args.wd

        self.optimizer = getattr(torch.optim, self.args.optimizer)
        self.optimizer = self.optimizer(self.model.parameters(), lr=self.lr, weight_decay=self.wd)
        
        self.criterion = self.set_criterion(self.args.criterion)
        self.scheduler = self.set_scheduler(args, self.optimizer)

        # self.__set_criterion(self.criterion)
        # self.__set_scheduler(self.args, self.optimizer)
        # self.cal = Calculate()
         
    def train(self, shuffle=True):
        for e in tqdm(range(self.epoch)):
            self.model.train()
            epoch_loss = 0
            losses = []

            for idx, x in enumerate(self.data_loader):
                x = x.float().to(device = self.device)
                if self.args.dataset == 'NAB':
                    x = x.reshape(x.shape[0], 1, -1)
                self.optimizer.zero_grad()
                
                recon_data, mu, log_var = self.model(x)
                # if self.args.dataset == 'NAB':
                #     x = x.reshape(x.shape[0], self.seq_len, -1)

                # print("x", x, "recon_data", recon_data)
                # loss = self.loss_fn(recon_data, x, mu, log_var)
                loss = self.criterion(x, recon_data) # reconstruction loss

                # mae = mean_absolute_error(x.flatten().cpu().detach().numpy(), pred.flatten().cpu().detach().numpy())
                # mse = mean_squared_error(x.flatten().cpu().detach().numpy(), pred.flatten().cpu().detach().numpy())
                
                interval = 300
                if (idx+1) % interval == 0:
                    print(f'[Epoch{e+1}] Loss:{loss}')
                # xs.extend(torch.mean(x, axis=(1,2)).cpu().detach().numpy().flatten()); preds.extend(torch.mean(pred, axis=(1,2)).cpu().detach().numpy().flatten())
                # maes.extend(mae.flatten()); mses.extend(mse.flatten())
                
                loss.backward()
                self.optimizer.step()
                # losses.append(loss.item())
                epoch_loss += loss

            
            torch.save(self.model.state_dict(), f'{self.args.save_path}model_{self.args.model}.pk')
            
            # wandb.log({"loss":loss})
            print(">>>", loss)
            # score = self.validation(0.94)
            if self.scheduler is not None:
                self.scheduler.step()

            # if best_score < score:
            #     print(f'Epoch : [{e}] Train loss : [{(epoch_loss/self.epoch)}] Val Score : [{score}])')
            #     best_score = score
            #     torch.save(self.model.module.state_dict(), f'./model_save/best_model_{1}.pth', _use_new_zipfile_serialization=False)
  
    
    def evaluation(self, test_loader, thr=0.95):
        cos = nn.CosineSimilarity(dim=1, eps=1e-6)
        self.model.eval()
        pred_list = []
        true_list = []
        diffs = []

        xs = []; preds = []; maes = []; mses = []; x_list = []; x_hat_list = []
        errors = []
        with torch.no_grad():
            for idx, (x, y) in enumerate(test_loader):
                x = x.float().to(device = self.device)
                self.optimizer.zero_grad()
                
                pred, mu, log_var = self.model(x)
                x_hat_list.append(pred)
                
                error = torch.sum(abs(x - pred), axis=(1,2))
                errors.extend(error)
              
                x_list.append(x); x_hat_list.append(pred)
                
                x = x.reshape(x.shape[0], -1)
                pred = pred.reshape(pred.shape[0], -1)

                y = y.reshape(y.shape[0], -1)
                y = y.mean(axis=1).numpy()
                y = np.where(y>0, 1, 0)
                true_list.extend(y.reshape(-1))
                
        errors = torch.tensor(errors, device = 'cpu').numpy()
        # errors_each = torch.tensor(errors_each, device='cpu').numpy()
        # x_real = torch.cat(x_list)
        # x_hat = torch.cat(x_hat_list)
        # x_real = x_real.cpu().detach().numpy()
        # x_hat = x_hat.cpu().detach().numpy()
        f1, precision, recall = self.get_metric(self.args.calc, self.args, true_list, 
                                                errors)


        # plt.cla()
        # plt.hist(diffs, bins=50, density=True, alpha=0.5) # histtype='stepfilled'
        # plt.title("Cosine similarity")
        # plt.savefig('Fig/cosine_similarity_difference.jpg')
        
        
        # plt.cla()
        # plt.figure(figsize=(15,8))
        # plt.ylim(0,0.5)
        # iidx = list(range(len(xs)))
        # plt.plot(iidx[:100], xs[:100], label="original")
        # plt.plot(iidx[:100], preds[:100], label="predict")
        # plt.fill_between(iidx[:100], xs[:100], preds[:100], color='green', alpha=0.5)
        # plt.title("Normal")
        # plt.legend()
        # plt.savefig(f'Fig/test_fill_between_AE_Normal.jpg')
        
        # plt.cla()
        # anomaly_idx = np.where(np.array(true_list) == 1)
        # anomaly_idx_bundle = find_bundle(anomaly_idx[0].tolist())
        # iidx = list(range(len(anomaly_idx_bundle[0])))
        # plt.plot(iidx[:-1], xs[anomaly_idx_bundle[0][0]:anomaly_idx_bundle[0][-1]], label="original")
        # plt.plot(iidx[:-1], preds[anomaly_idx_bundle[0][0]:anomaly_idx_bundle[0][-1]], label="predict")
        # plt.fill_between(iidx[:-1], xs[anomaly_idx_bundle[0][0]:anomaly_idx_bundle[0][-1]], preds[anomaly_idx_bundle[0][0]:anomaly_idx_bundle[0][-1]], color='green', alpha=0.5)
        # plt.title("Abnormal")
        # plt.legend()
        # plt.savefig(f'Fig/test_fill_between_LSTMAE_Anomaly.jpg')

        # plt.cla()
        # plt.hist(mses, density=True, bins=100, alpha=0.5)
        # plt.xlim(0,0.1)
        # plt.savefig(f'Fig/test_distribution_LSTMAE_mse.jpg')

        # plt.cla()
        # plt.hist(maes, density=True, bins=100, alpha=0.5)
        # plt.xlim(0,0.2)
        # plt.savefig(f'Fig/test_distribution_LSTMAE_mae.jpg')

        # thres = np.percentile(np.array(maes), 99)
        # pred_list = np.where(np.array(maes)>thres, 1, 0)
        # f1 = f1_score(true_list, pred_list, average='macro')
        # precision = precision_score(true_list, pred_list, average="macro")
        # recall = recall_score(true_list, pred_list, average="macro")
        # print(f"f1 score {f1}")

        # print(confusion_matrix(true_list, pred_list))
        return f1, precision, recall

    def loss_fn(self, recon_x, x, mu, log_var):
        BCE = nn.functional.mse_loss(recon_x, x, reduction='sum')
        KLD = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
        return BCE + KLD

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

    def get_score(self, method, true_list, errors):
        from Score import make_pred
        score_func = make_pred.Pred_making()

        if method == 'quantile':
            true_list, pred_list = score_func.quantile_score(true_list, errors)
        return true_list, pred_list
    
    def get_metric(self, method, args, true_list, errors, 
                   true_list_each=None, errors_each=None):
        from Score import make_pred
        from Score.calculate_score import Calculate_score
        score_func = make_pred.Pred_making()
        metric_func = Calculate_score(args)
        
        if method == 'default':
            true_list, pred_list = score_func.quantile_score(true_list, errors)
            f1, precision, recall = metric_func.score(true_list, pred_list)
        elif method == 'fix':
            true_list, pred_list = score_func.fix_threshold(true_list, errors)
            f1, precision, recall = metric_func.score(true_list, pred_list)
        elif method == 'back':
            true_list, pred_list = score_func.quantile_score(true_list_each, errors_each)
            f1, precision, recall = metric_func.back_score(true_list, pred_list)
        return f1, precision, recall
