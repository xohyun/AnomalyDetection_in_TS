import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import tqdm
import wandb
from tqdm import tqdm
from sklearn.metrics import f1_score
from utils.utils import gpu_checking
from Trainer.base_trainer import base_trainer

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
        self.seq_len = data_info['seq_len']
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
            l1s = []
            l2s = []
            self.model.train()
            
            for idx, x in enumerate(self.data_loader):
                x = x.float().to(self.device)
                self.optimizer.zero_grad()
                # x = x.reshape() # reshape
                
                ae1s, ae2s, ae2ae1s = self.model(x.to(device=self.device))
                x = torch.flatten(x)
                l1 = (1/(e+1)) * self.criterion(ae1s, x) + (1-1/(e+1)) * self.criterion(ae2ae1s, x)
                l2 = (1/(e+1)) * self.criterion(ae2s, x) + (1-1/(e+1)) * self.criterion(ae2ae1s, x)
                l1s.append(torch.mean(l1).item())
                l2s.append(torch.mean(l2).item())
                loss = torch.mean(l1 + l2)
                # if (idx+1) % 1000 == 0:
                #     print("1000th")
                    # print(f"[Epoch{e+1}, Step({idx+1}/{len(self.data_loader.dataset)}), Loss:{:/4f}")

                loss.backward()
                self.optimizer.step()
                
            # wandb.log({"l1":np.mean(l1s), "l2":np.mean(l2s), "loss":loss})
            torch.save(self.model.state_dict(), f'{self.args.save_path}model_{self.args.model}.pk')
            
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
            ae1s, ae2s, ae2ae1s = [], [], []

            for idx, (x, y) in enumerate(test_loader):
                x = x.float().to(device = self.device)
                self.optimizer.zero_grad()
                
                ae1, ae2, ae2ae1 = self.model(x)
                ae1s.append(ae1); ae2s.append(ae2); ae2ae1s.append(ae2ae1)
                # x_hat_list.append(pred)
                
                # error = torch.sum(abs(x - pred), axis=(1,2))
                # errors.extend(error)
              
                # x_list.append(x); x_hat_list.append(pred)
                xs.append(x)
                x = x.reshape(x.shape[0], -1)
                pred = pred.reshape(pred.shape[0], -1)

                y = y.reshape(y.shape[0], -1)
                y = y.mean(axis=1).numpy()
                y = np.where(y>0, 1, 0)
                true_list.extend(y.reshape(-1))
            
        ae1s, ae2s, ae2ae1s = torch.stack(ae1s), torch.stack(ae2s), torch.stack(ae2ae1s)
        x = torch.stack(xs)
        y_pred = ae1s[:, x.shape[1]-self.features:x.shape[1]].view(-1, self.features)

        errors = torch.sum(abs(x - y_pred), axis=(1,2))
        errors = torch.tensor(errors, device = 'cpu').numpy()
        # errors_each = torch.tensor(errors_each, device='cpu').numpy()
        # x_real = torch.cat(x_list)
        # x_hat = torch.cat(x_hat_list)
        # x_real = x_real.cpu().detach().numpy()
        # x_hat = x_hat.cpu().detach().numpy()
        f1, precision, recall = self.get_metric(self.args.calc, self.args, true_list, 
                                                errors)

        return f1, precision, recall


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