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
from utils.feature_extractor import FeatureExtractor
from utils.metrics import mahalanobis

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
                
                feature, pred = self.model(x)
                loss = self.criterion(x, pred)

                # mae = mean_absolute_error(x.flatten().cpu().detach().numpy(), pred.flatten().cpu().detach().numpy())
                # mse = mean_squared_error(x.flatten().cpu().detach().numpy(), pred.flatten().cpu().detach().numpy())

            
                xs.extend(torch.mean(x, axis=(1,2)).cpu().detach().numpy().flatten()); preds.extend(torch.mean(pred, axis=(1,2)).cpu().detach().numpy().flatten())
                # maes.extend(mae.flatten()); mses.extend(mse.flatten())
                interval = 300
                if (idx+1) % interval == 0: print(f'[Epoch{e+1}] Loss:{loss}')

                loss.backward()
                self.optimizer.step()
                epoch_loss += loss
            
            torch.save(self.model.state_dict(), f'{self.args.save_path}model_{self.args.model}.pk')
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
        pred_list = []
        true_list = []
        diffs = []
        test_embeds = None

        flag = list(self.model._modules)[-1]
        final_layer = self.model._modules.get(flag)
        activated_features = FeatureExtractor(final_layer)

        xs = []; preds = []; maes = []; mses = []; x_list = []; x_hat_list = []; errors = []
        features = []
        with torch.no_grad():
            for idx, (x, y) in enumerate(test_loader):
                x = x.float().to(device = self.device)
                self.optimizer.zero_grad()
                batch = x.shape[0]

                feature, pred = self.model(x)
                features.extend(feature)
                # error = torch.sum(abs(x - pred), axis=(1,2)).cpu().detach()
                # errors.extend(error)
                
                # mae = mean_absolute_error(x.flatten().cpu().detach().numpy(), pred.flatten().cpu().detach().numpy())
                # mse = mean_squared_error(x.flatten().cpu().detach().numpy(), pred.flatten().cpu().detach().numpy())
                
                # mae = mean_absolute_error(np.transpose(x.reshape(batch, -1).cpu().detach().numpy()), np.transpose(pred.reshape(batch, -1).cpu().detach().numpy()), multioutput='raw_values')
                # mse = mean_squared_error(np.transpose(x.reshape(batch, -1).cpu().detach().numpy()), np.transpose(pred.reshape(batch, -1).cpu().detach().numpy()), multioutput='raw_values')

                x_list.append(x); x_hat_list.append(pred)
                # xs.extend(torch.mean(x, axis=(1,2)).cpu().detach().numpy().flatten()); preds.extend(torch.mean(pred, axis=(1,2)).cpu().detach().numpy().flatten())
                # maes.extend(mae.flatten()); mses.extend(mse.flatten())

                x = x.reshape(x.shape[0], -1)
                pred = pred.reshape(pred.shape[0], -1)
                # diff = cos(x, pred).cpu().tolist()
                
                # batch_pred = np.where(((np.array(diff)<0) & (np.array(diff)>-0.1)), 1, 0)
                # batch_pred = np.where(np.array(diff)<0.7, 1, 0)
                # y = np.where(y>0.69, 1, 0)
            
                # diffs.extend(diff)
                # pred_list.extend(batch_pred)
                y = y.reshape(y.shape[0], -1)
                y = y.mean(axis=1).numpy()
                y = np.where(y>0, 1, 0)
                true_list.extend(y)

                #---# for t-sne #---#
                # embeds = activated_features.features_before # [256, 200, 1, 4]; embeds =  embeds.squeeze(2) # [256, 200, 4]
                # if test_embeds == None: test_embeds = embeds
                # else : test_embeds = torch.cat((test_embeds, embeds), dim=0) # [103, 800]

        features = torch.vstack(features)
        print(">>>", featuers.shape)
        # x_real = torch.cat(x_list)
        # x_hat = torch.cat(x_hat_list)
        # x_real = x_real.cpu().detach().numpy()
        # x_hat = x_hat.cpu().detach().numpy()



        # l_quantile = np.quantile(np.array(errors), 0)
        # u_quantile = np.quantile(np.array(errors), 0.96)
        # in_range = np.logical_and(np.array(errors) >= l_quantile, np.array(errors) <= u_quantile)
        # # pred_list = [0 for i in errors if i in in_range]
        # np_errors = np.array(errors)
        # # pred_list = [i for i in np_errors if i in np.where((i >= l_quantile and i <= u_quantile), 0, 1)]
        # pred_list = np.zeros(len(errors))
        # for i in range(len(np_errors)):
        #     if errors[i] >= l_quantile and errors[i] <= u_quantile:
        #         pred_list[i] = 0
        #     else:
        #         pred_list[i] = 1

        # np.savez(f"./features_", test_embeds=test_embeds, true_list=true_list)
        # errors, predictions_vs = reconstruction_errors(x_real, x_hat, score_window=self.args.seq_len, step_size=1) #score_window=config.window_size
        # error_range = find_anomalies(errors, index=range(len(errors)), anomaly_padding=5)
        # pred_list = np.zeros(len(true_list))
        # for i in error_range:
        #     start = int(i[0])
        #     end = int(i[1])
        #     pred_list[start:end] = 1

        # drawing(config, anomaly_value, pd.DataFrame(test_dataset.scaled_test))
        
        raise
        x_real = x_real.flatten()
        x_hat = x_hat.flatten()

        f1 = f1_score(true_list, pred_list, average='macro')
        precision = precision_score(true_list, pred_list, average="macro")
        recall = recall_score(true_list, pred_list, average="macro")
        
        print(confusion_matrix(true_list, pred_list))
        print(f"f1 {f1} / precision {precision} / recall {recall}")
        
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
        # plt.savefig(f'Fig/test_fill_between_AE_Anomaly.jpg')


        # plt.cla()
        # plt.hist(mses, density=True, bins=100, alpha=0.5)
        # plt.xlim(0,0.1)
        # plt.savefig(f'Fig/test_distribution_AE_mse.jpg')

        # plt.cla()
        # plt.hist(maes, density=True, bins=100, alpha=0.5)
        # plt.xlim(0,0.2)
        # plt.savefig(f'Fig/test_distribution_AE_mae.jpg')

        # thres = np.percentile(np.array(maes), 99)
        # pred_list = np.where(np.array(maes)>thres, 1, 0)
        # f1 = f1_score(true_list, pred_list, average='macro')
        # precision = precision_score(true_list, pred_list, average="macro")
        # recall = recall_score(true_list, pred_list, average="macro")
        # print(f"f1 score {f1}")

        # print(confusion_matrix(true_list, pred_list))
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