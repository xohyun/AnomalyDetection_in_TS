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

        self.device = gpu_checking(args)

        self.features = data_info['num_features']
        self.epoch = self.args.epoch

        self.lr = self.args.lr
        self.wd = self.args.wd

        self.optimizer = getattr(torch.optim, self.args.optimizer)
        self.optimizer = self.optimizer(
            self.model.parameters(), lr=self.lr, weight_decay=self.wd)

        self.criterion = self.set_criterion(self.args.criterion)
        self.scheduler = self.set_scheduler(args, self.optimizer)

        # self.__set_criterion(self.criterion)
        # self.__set_scheduler(self.args, self.optimizer)
        # self.cal = Calculate()

    def train(self, shuffle=True):
        for e in tqdm(range(self.epoch)):
            epoch_loss = 0
            self.model.train()
            xs = []
            preds = []
            maes = []
            mses = []

            for idx, x in enumerate(self.data_loader):
                x = x.float().to(device=self.device)
                self.optimizer.zero_grad()

                forecast_part = x[:, int(self.args.seq_len*0.8):, :]
                variances = torch.var(forecast_part, dim=1)

                output = self.model(x)

                pred = torch.concat(
                    (output["reconstructs"], output["forecasts"]), dim=1)
                loss = self.criterion(x, pred)
                loss_var = self.criterion(variances, output["variances"])

                loss = loss + loss_var
                # mae = mean_absolute_error(x.flatten().cpu().detach().numpy(), pred.flatten().cpu().detach().numpy())
                # mse = mean_squared_error(x.flatten().cpu().detach().numpy(), pred.flatten().cpu().detach().numpy())

                xs.extend(torch.mean(x, axis=(1, 2)
                                     ).cpu().detach().numpy().flatten())
                preds.extend(torch.mean(pred, axis=(1, 2)
                                        ).cpu().detach().numpy().flatten())
                # maes.extend(mae.flatten()); mses.extend(mse.flatten())
                interval = 300
                if (idx+1) % interval == 0:
                    print(f'[Epoch{e+1}] Loss:{loss}')

                loss.backward()
                self.optimizer.step()
                epoch_loss += loss

            torch.save(self.model.state_dict(),
                       f'{self.args.save_path}model_{self.args.model}.pk')
            print(f"{e}epoch / loss {loss}")

            # wandb.log({"loss":loss})
            # score = self.validation(0.94)
            if self.scheduler is not None:
                self.scheduler.step()
            # iidx = list(range(len(xs)))

            # plt.figure(figsize=(15,8))
            # plt.ylim(0,0.5)
            # plt.plot(iidx[:100], xs[:100], label="original")
            # plt.plot(iidx[:100], preds[:100], label="predict")
            # plt.fill_between(iidx[:100], xs[:100], preds[:100], color='green', alpha=0.5)
            # plt.legend()
            # plt.savefig(f'Fig/train_fill_between_AE_{e}.jpg')

            # plt.cla()
            # plt.hist(mses, bins=100, density=True, alpha=0.5)
            # plt.xlim(0,0.1)
            # plt.savefig(f'Fig/train_distribution_AE_mse.jpg')

            # plt.cla()
            # plt.hist(maes, bins=100, density=True, alpha=0.5)
            # plt.xlim(0,0.2)
            # plt.savefig(f'Fig/train_distribution_AE_mae.jpg')

    def evaluation(self, test_loader, thr=0.95):
        cos = nn.CosineSimilarity(dim=1, eps=1e-6)
        self.model.eval()
        true_list = []
        true_list_each = []
        diffs = []

        xs = []
        preds = []
        maes = []
        mses = []
        x_list = []
        x_hat_list = []
        errors = []

        with torch.no_grad():
            for idx, (x, y) in enumerate(test_loader):
                x = x.float().to(device=self.device)
                self.optimizer.zero_grad()
                batch = x.shape[0]

                forecast_part = x[:, int(self.args.seq_len*0.8):, :]
                variances = torch.var(forecast_part, dim=1)

                output = self.model(x)
                pred = torch.concat((output["reconstructs"], output["forecasts"]), dim=1)

                error = torch.sum(abs(x - pred), axis=(1, 2)).cpu().detach()
                errors.extend(error)

                x_list.append(x)
                x_hat_list.append(pred)
                
                x = x.reshape(x.shape[0], -1)
                pred = pred.reshape(pred.shape[0], -1)
                
                y = y.reshape(y.shape[0], -1)
                true_list_each.extend(y)
                y = y.mean(axis=1).numpy()
                y = np.where(y > 0, 1, 0)
                true_list.extend(y)

        x_real = torch.cat(x_list)
        x_hat = torch.cat(x_hat_list)
        x_real = x_real.cpu().detach().numpy()
        x_hat = x_hat.cpu().detach().numpy()

        # x_real = x_real.flatten()
        # x_hat = x_hat.flatten()
        from Score.PA_back import PA_back
        scoring = PA_back()
        # scoring = self.get_score(self.args.score)
        print("???", len(true_list_each))
        f1, precision, recall = scoring.score(true_list_each, errors)       
        # return f1, precision, recall

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
            scheduler = optim.lr_scheduler.ExponentialLR(
                optimizer, gamma=args.gamma)
        elif args.scheduler == 'step':
            scheduler = optim.lr_scheduler.StepLR(
                optimizer, step_size=args.step_size, gamma=args.gamma)
        elif args.scheduler == 'multi_step':
            scheduler = optim.lr_scheduler.MultiStepLR(
                optimizer, milestones=args.milestones, gamma=args.gamma)
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

    def get_score(self, method):
        if method == 'PA':
            from Score import PA
            score_fun = PA.PA()
        return score_fun