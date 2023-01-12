import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import tqdm
from tqdm import tqdm
from sklearn.metrics import f1_score
from utils import gpu_checking

class TrainMaker:
    def __init__(self, args, model, data_loaders, data_info):
        self.args = args
        self.model = model
        self.data_loader = data_loaders['train']
        self.data_loader_valid = data_loaders['valid']
  
        self.device =  gpu_checking(args)

        self.features = data_info
        self.epoch = self.args.epoch
        
        self.lr = self.args.lr
        self.wd = self.args.wd
        
        # 둘중에 뭐가맞지?
        self.optimizer = getattr(torch.optim, self.args.optimizer)
        # self.optimizer = self.optimizer(self.model.parameters(), lr=self.lr, weight_decay=self.wd)

        self.criterion = self.__set_criterion(self.args.criterion)
        self.scheduler = self.__set_scheduler(args, self.optimizer)
        
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

                if (idx+1) % 1000 == 0:
                    print("1000th")
                    # print(f"[Epoch{e+1}, Step({idx+1}/{len(self.data_loader.dataset)}), Loss:{:/4f}")

                self.loss.backward()
                self.optimizer.step()
                epoch_loss += loss
            
            score = self.validation(0.94)

            if self.scheduler is not None:
                self.scheduler.step(score)

            if best_score < score:
                print(f'Epoch : [{e}] Train loss : [{(epoch_loss/self.epoch)}] Val Score : [{score}])')
                best_score = score
                torch.save(self.model.module.state_dict(), f'./model_save/best_model_{1}.pth', _use_new_zipfile_serialization=False)
  
    
    def validation(self, thr):
        cos = nn.CosineSimilarity(dim=1, eps=1e-6)
        self.model.eval()
        pred = []
        true = []
        with torch.no_grad():
            for x, y in iter(self.val_loader):
                x = x.float().to(self.device)

                _x = self.model(x)
                diff = cos(x, _x).cpu().tolist()
                batch_pred = np.where(np.array(diff)<thr, 1,0).tolist()
                pred += batch_pred
                true += y.tolist()

        return f1_score(true, pred, average='macro')


    def __set_criterion(self, criterion):
        if criterion == "MSE":
            criterion = nn.MSELoss()
        elif criterion == "CEE":
            criterion = nn.CrossEntropyLoss()
        elif criterion == "cosine":
            criterion = nn.CosineEmbeddingLoss()
            
        return criterion
    
    def __set_scheduler(self, args, optimizer):
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