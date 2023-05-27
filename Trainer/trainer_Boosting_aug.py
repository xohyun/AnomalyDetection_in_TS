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
            
            for idx, x in enumerate(self.data_loader):
                x = x.float().to(device=self.device)
                self.optimizer.zero_grad()

                forecast_part = x[:, int(self.args.seq_len*self.args.recon_ratio):, :]
                variances = torch.var(forecast_part, dim=1)

                output = self.model(x)
                output["real_variances"] = variances

                pred = torch.concat(
                    (output["reconstructs"], output["forecasts"]), dim=1)
                loss = self.criterion(x, pred)
                loss_var = self.criterion(variances, output["variances"])

                #---# sum of variance loss and CEE #---#
                loss = 0.8*loss + 0.2*loss_var # can change ratio # alpha, beta

                #---# Print loss #---#
                interval = 300
                if (idx+1) % interval == 0:
                    print(f'[Epoch{e+1}] Loss:{loss}')

                loss.backward()
                self.optimizer.step()
                epoch_loss += loss

            #---# Model save #---#
            torch.save(self.model.state_dict(),
                       f'{self.args.save_path}model_{self.args.model}.pk')
            print(f"{e}epoch / loss {loss}")
            # wandb.log({"loss":loss})

            if self.scheduler is not None:
                self.scheduler.step()

        # ---# To save trainer data #---#
        outputs = []
        with torch.no_grad():
            for idx, x in enumerate(self.data_loader):
                x = x.float().to(device=self.device)
                output = self.model(x)
                output['x'] = x
                outputs.append(output)
            outputs = np.array(outputs)
            np.save(f'{self.args.save_path}train_output.npy', outputs)

    def evaluation(self, test_loader):
        cos = nn.CosineSimilarity(dim=1, eps=1e-6)
        self.model.eval()

        dist_list = []
        x_list = []; true_list = []; true_list_each = []
        x_hat_list = []
        errors = []; errors_each = []
        xs = []; preds = []

        #---# For feature extractor #---#
        from utils.feature_extractor import FeatureExtractor
        flag = list(self.model._modules)[-1] #################
        final_layer = self.model._modules.get(flag)
        activated_features = FeatureExtractor(final_layer) #############################

        recon_var_list = []; fore_var_list = []; 
        test_embeds = None
        with torch.no_grad():
            for idx, (x, y) in enumerate(test_loader):
                x = x.float().to(device=self.device)
                batch = x.shape[0]
                # plt.figure(figsize=(30,8))
                # plt.plot(x.reshape(-1,x.shape[2]).detach().cpu().numpy()[:,0])
                # plt.savefig("ddddddd.png")
                
                self.optimizer.zero_grad()
                forecast_part = x[:, int(self.args.seq_len*self.args.recon_ratio):, :]
                variances = torch.var(forecast_part, dim=1)

                output = self.model(x) # dictionary format
                pred = torch.concat(
                    (output["reconstructs"], output["forecasts"]), dim=1) # to make pred
                xs.append(x); preds.append(pred)
                
                # var_dist
                pred_var = torch.var(pred, dim=1)
                recon_var_list.append(output["variances"]) # reconstruct variance
                fore_var_list.append(pred_var) # forecast variance
                
                error = torch.sum(abs(x - pred), axis=(1, 2)) # for every batch
                errors.extend(error)
                errors_each.append(abs(x - pred)) # no sum

                x_list.append(x)
                x_hat_list.append(pred)

                x = x.reshape(x.shape[0], -1)
                pred = pred.reshape(pred.shape[0], -1)

                true_list_each.extend(y.reshape(-1))
                y = y.reshape(y.shape[0], -1)
                y = y.mean(axis=1).numpy()
                y = np.where(y > 0, 1, 0)
                true_list.extend(y)

                #---# For t-SNE #---#
                embeds = activated_features.features_before # [256, 200, 1, 4]; embeds =  embeds.squeeze(2) # [256, 200, 4]
                if test_embeds == None: test_embeds = embeds
                else : test_embeds = torch.cat((test_embeds, embeds), dim=0) # [103, 800]


        dist_list = {"recon_var_list":recon_var_list, "fore_var_list":fore_var_list}
        errors = torch.tensor(errors, device='cpu').numpy()
        errors_each = torch.cat(errors_each)
        errors_each = errors_each.clone().detach().cpu().numpy() # to numpy

        #---# forecast drawing #---#
        xs_ = torch.cat(xs).clone().detach().cpu().numpy()
        xs_ = xs_.reshape(-1, xs_.shape[2])
        preds_ = torch.cat(preds).clone().detach().cpu().numpy()
        preds_ = preds_.reshape(-1, preds_.shape[2])
        plt.figure(figsize=(30,8))
        plt.plot(xs_[21000:25000,0])
        plt.savefig('original.png')
        plt.clf()
        for i in range(23600,23900,50):
            idx = i
            show_len = 70
            plt.figure(figsize=(20,8))
            plt.ylim((-2,2))
            temp = np.concatenate((xs_[idx:idx+show_len,0], preds_[idx+show_len:idx+show_len+13, 0]))
            plt.plot(temp, label='Forecast', color='orange')
            plt.plot(xs_[idx:idx+show_len+13, 0], label='Actual', color='tab:blue')
            
            plt.legend(fontsize=20)
            plt.savefig(f'./forecast{idx}.png')

        true_list, pred_list = self.get_score(self.args, self.args.score, true_list, errors, dist_list, errors_each)
        f1, precision, recall = self.get_metric(self.args.calc, self.args, true_list, 
                                                pred_list, true_list_each, errors_each)
        mae, rmse, mape = self.get_score_forecast(self.args, xs, preds)
        print(f"mae : {mae} / rmse : {rmse} / mape : {mape}")

        #---# For save features #---#                                      
        # np.savez(f"./features/features_", test_embeds=test_embeds, true_list=true_list)

        return f1, precision, recall, mae, rmse, mape

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

    def get_score(self, args, method, true_list, errors, dist_list=None, errors_each=None):
        from Score import make_pred
        score_func = make_pred.Pred_making()

        if method == 'quantile':
            true_list, pred_list = score_func.quantile_score(true_list, errors)
        elif method == 'variance':
            true_list, pred_list = score_func.variance_score(true_list, errors, dist_list)
        elif method == 'var_weight':
            true_list, pred_list = score_func.variance_score_with_weighted_sum(true_list, errors_each, dist_list)
        elif method == 'var_corr':
            true_list, pred_list = score_func.variance_score_with_corr(args, true_list, errors_each, dist_list)
        return true_list, pred_list
    
    def get_score_forecast(self, args, xs, preds):
        from utils.metrics import MAE, RMSE, MAPE
        preds = torch.cat(preds).clone().detach().cpu().numpy()
        xs = torch.cat(xs).clone().detach().cpu().numpy()
        
        pred = preds.reshape(-1, preds.shape[2])[:,0]
        true = xs.reshape(-1, xs.shape[2])[:,0]
        mae = MAE(pred, true)
        rmse = RMSE(pred, true)
        mape = MAPE(pred, true)
        return mae, rmse, mape

    def get_metric(self, method, args, true_list, pred_list,
                   true_list_each=None, errors_each=None):
        from Score.calculate_score import Calculate_score
        metric_func = Calculate_score(args)

        if method == 'default':
            f1, precision, recall = metric_func.score(true_list, pred_list)
        elif method == 'back':
            f1, precision, recall = metric_func.back_score(true_list, pred_list)
       
        return f1, precision, recall