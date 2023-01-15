import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import tqdm
from tqdm import tqdm
from sklearn.metrics import f1_score
from utils import gpu_checking
from Trainer.base_trainer import base_trainer

class TrainMaker(base_trainer):
    def __init__(self, args, model, data_loaders, data_info):
        self.args = args
        self.model = model
        self.mode = self.args.mode

        if self.mode == "train":
            self.data_loader = data_loaders['train']
            self.data_loader_valid = data_loaders['valid']
        else:
            self.data_loader_test = data_loaders['test']
  
        self.device =  gpu_checking(args)

        self.features = data_info
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
            
            for idx, x in enumerate(self.data_loader):
                x = x.float().to(self.device)
                self.optimizer.zero_grad()
                # x = x.reshape() # reshape
                
                pred = self.model(x.to(device=self.device))
                loss = self.criterion(x, pred)

                # if (idx+1) % 1000 == 0:
                #     print("1000th")
                    # print(f"[Epoch{e+1}, Step({idx+1}/{len(self.data_loader.dataset)}), Loss:{:/4f}")

                loss.backward()
                self.optimizer.step()
                epoch_loss += loss
            
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
        with torch.no_grad():
            for idx, (x, y) in enumerate(test_loader):
                x = x.float().to(self.device)    
                pred = self.model(x.to(device=self.device))

                x = x.reshape(x.shape[0], -1)
                pred = pred.reshape(pred.shape[0], -1)
                y = y.reshape(y.shape[0], -1)
                y = y.mean(axis=1).numpy()
                # print(y.shape, "====")
                diff = cos(x, pred).cpu().tolist()

                # print(diff)
                batch_pred = np.where(abs(np.array(diff))<0.001, 1, 0)
                y = np.where(y>0.69, 1, 0)
                
                # print(batch_pred.shape)

                pred_list.extend(batch_pred)
                true_list.extend(y)

        f1 = f1_score(true_list, pred_list, average='macro')
        print(f"f1 score {f1}")

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