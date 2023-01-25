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
            epoch_loss = 0
            self.model.train()
            xs = []; preds = []; maes = []; mses = []
            for idx, x in enumerate(self.data_loader):
                x = x.float().to(device = self.device)
                self.optimizer.zero_grad()
                
                pred = self.model(x)
                loss = self.criterion(x, pred)

                mae = mean_absolute_error(x.flatten().detach().numpy(), pred.flatten().detach().numpy())
                mse = mean_squared_error(x.flatten().detach().numpy(), pred.flatten().detach().numpy())

                # if (idx+1) % 1000 == 0:
                #     print("1000th")
                    # print(f"[Epoch{e+1}, Step({idx+1}/{len(self.data_loader.dataset)}), Loss:{:/4f}")

                xs.extend(x.flatten()); preds.append(pred.flatten())
                maes.extend(mae.flatten()); mses.append(mse.flatten())
                
                loss.backward()
                self.optimizer.step()
                epoch_loss += loss
            
            wandb.log({"loss":loss})
            # score = self.validation(0.94)
            if self.scheduler is not None:
                self.scheduler.step()
            
            plt.fill_between(xs, preds, bins=50, density=True, alpha=0.5, histtype='stepfilled')
            plt.savefig(f'/Fig/train_fill_between_AE_{e}.jpg')
            plt.hist(mse, bins=50, density=True, alpha=0.5, histtype='stepfilled')
            plt.hist(mae, bins=50, density=True, alpha=0.5, histtype='stepfilled')
            plt.savefig(f'/Fig/train_distribution_AE_{e}.jpg')
            

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

        xs = []; preds = []; maes = []; mses = []
        with torch.no_grad():
            for idx, (x, y) in enumerate(test_loader):
                x = x.float().to(self.device)    
                pred = self.model(x.to(device=self.device))

                x = x.reshape(x.shape[0], -1)
                pred = pred.reshape(pred.shape[0], -1)
                
                mae = mean_absolute_error(x.flatten().detach().numpy(), pred.flatten().detach().numpy())
                mse = mean_squared_error(x.flatten().detach().numpy(), pred.flatten().detach().numpy())

                xs.append(x); preds.append(pred)
                maes.append(mae); mses.append(mse)

                y = y.reshape(y.shape[0], -1)
                y = y.mean(axis=1).numpy()

                diff = cos(x, pred).cpu().tolist()
                
                # batch_pred = np.where(((np.array(diff)<0) & (np.array(diff)>-0.1)), 1, 0)
                batch_pred = np.where(abs(np.array(diff))<0.01, 1, 0)
                # y = np.where(y>0.69, 1, 0)
                y = np.where(y>0, 1, 0)

                diffs.extend(diff)
                pred_list.extend(batch_pred)
                true_list.extend(y)
        

        plt.hist(diffs, bins=50, density=True, alpha=0.5, histtype='stepfilled')
        plt.savefig('/Fig/cosine_similarity_difference.jpg')
        f1 = f1_score(true_list, pred_list, average='macro')
        
        plt.fill_between(xs.flatten(), preds.flatten(), bins=50, density=True, alpha=0.5, histtype='stepfilled')
        plt.savefig(f'/Fig/test_fill_between_AE.jpg')
        plt.hist(mse, bins=50, density=True, alpha=0.5, histtype='stepfilled')
        plt.hist(mae, bins=50, density=True, alpha=0.5, histtype='stepfilled')
        plt.savefig(f'/Fig/test_distribution_AE.jpg')
        
        print(f"f1 score {f1}")
        print(confusion_matrix(true_list, pred_list))
        return f1


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