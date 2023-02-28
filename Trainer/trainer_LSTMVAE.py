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
            epoch_loss = 0
            self.model.train()
            xs = []; preds = []; maes = []; mses = []
            
            hidden = self.model.init_hidden(self.args.batch_size)

            for idx, (x, index_x) in enumerate(self.data_loader):
                batch = x.shape[0]
                targetSeq = self.get_batch(index_x, self.data_loader) # shape : [seq_len, batch, features]
                
                x = x.reshape(targetSeq.shape)
                hidden = self.model.repackage_hidden(hidden) 
                hidden_ = self.model.repackage_hidden(hidden)

                x = x.float().to(device = self.device)
                self.optimizer.zero_grad()
                
                # pred = self.model(x, hidden)

                '''Loss1: Free running loss'''
                outVal = x[0].unsqueeze(0)
                outVals=[]
                hids1 = []
                for i in range(x.size(0)):
                    outVal, hidden_, hid = self.model.forward(outVal, hidden_,return_hiddens=True)
                    outVals.append(outVal)
                    hids1.append(hid)
                
                outSeq1 = torch.cat(outVals,dim=0)
                hids1 = torch.cat(hids1,dim=0)
                loss1 = self.criterion(outSeq1.reshape(batch,-1), targetSeq.reshape(batch,-1))

                '''Loss2: Teacher forcing loss'''
                outSeq2, hidden, hids2 = self.model.forward(x, hidden, return_hiddens=True)
                loss2 = self.criterion(outSeq2.view(batch, -1), targetSeq.view(batch, -1))

                '''Loss3: Simplified Professor forcing loss'''
                loss3 = self.criterion(hids1.view(batch,-1), hids2.view(batch, -1).detach())

                '''Total loss = Loss1+Loss2+Loss3'''
                loss = loss1+loss2+loss3
                loss.backward()



                # mae = mean_absolute_error(x.flatten().cpu().detach().numpy(), pred.flatten().cpu().detach().numpy())
                # mse = mean_squared_error(x.flatten().cpu().detach().numpy(), pred.flatten().cpu().detach().numpy())
    
        
                # maes.extend(mae.flatten()); mses.extend(mse.flatten())
                interval = 300
                if (idx+1) % interval == 0: print(f'[Epoch{e+1}] Loss:{loss}')

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

        xs = []; preds = []; maes = []; mses = []; x_list = []; x_hat_list = []; errors = []
        train_dataset = self.get_trainset(self.args)
        channel_idx = 0
        mean, cov = self.fit_norm_distribution_param(self.args, self.model, train_dataset, channel_idx=channel_idx)
        
        with torch.no_grad():
            for idx, (x, y) in enumerate(test_loader):
                x = x.float().to(device = self.device)
                self.optimizer.zero_grad()
                batch = x.shape[0]

                pred = self.model(x)
                
                error = torch.sum(abs(x - pred), axis=(1,2)).cpu().detach()
                errors.extend(error)
                
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
        
        
        score, sorted_prediction, sorted_error, _, predicted_score = self.anomalyScore(self.args, self.model, test_dataset, mean, cov,
                                                                                  score_predictor=score_predictor,
                                                                                  channel_idx=channel_idx)
        
        precision, recall, f_beta = self.get_precision_recall(args, score, num_samples=1000, beta=args.beta,
                                                         label=TimeseriesData.testLabel.to(args.device))
    
        # return f1, precision, recall
        return f_beta, precision, recall

    def get_batch(self, idx, train_loader):
        next_idx = idx + 1
        next_seq = train_loader.dataset.data_x[next_idx]
        next_seq = next_seq.reshape(next_seq.shape[1], next_seq.shape[0], -1)
        next_seq = torch.tensor(next_seq).float().to(device = self.device)
        return next_seq
    
    def get_trainset(self, args):
        _args = args
        _args.mode = "train"
        from DataLoader.data_provider import get_dataloader
        dl = get_dataloader(_args)
        return dl.data_loaders.dataset

    def fit_norm_distribution_param(self, args, model, train_dataset, channel_idx=0):
        predictions = []
        organized = []
        errors = []
        with torch.no_grad():
            # Turn on evaluation mode which disables dropout.
            model.eval()
            pasthidden = model.init_hidden(1)
            for t in range(len(train_dataset)):
                out, hidden = model.forward(train_dataset[t].unsqueeze(0), pasthidden)
                predictions.append([])
                organized.append([])
                errors.append([])
                predictions[t].append(out.data.cpu()[0][0][channel_idx])
                pasthidden = model.repackage_hidden(hidden)
                for prediction_step in range(1,args.prediction_window_size):
                    out, hidden = model.forward(out, hidden)
                    predictions[t].append(out.data.cpu()[0][0][channel_idx])

                if t >= args.prediction_window_size:
                    for step in range(args.prediction_window_size):
                        organized[t].append(predictions[step+t-args.prediction_window_size][args.prediction_window_size-1-step])
                    organized[t]= torch.FloatTensor(organized[t]).to(args.device)
                    errors[t] = organized[t] - train_dataset[t][0][channel_idx]
                    errors[t] = errors[t].unsqueeze(0)

        errors_tensor = torch.cat(errors[args.prediction_window_size:],dim=0)
        mean = errors_tensor.mean(dim=0)
        cov = errors_tensor.t().mm(errors_tensor)/errors_tensor.size(0) - mean.unsqueeze(1).mm(mean.unsqueeze(0))
        # cov: positive-semidefinite and symmetric.

        return mean, cov

    def anomalyScore(self, args, model, dataset, mean, cov, channel_idx=0, score_predictor=None):
        predictions = []
        rearranged = []
        errors = []
        hiddens = []
        predicted_scores = []
        with torch.no_grad():
            # Turn on evaluation mode which disables dropout.
            model.eval()
            pasthidden = model.init_hidden(1)
            for t in range(len(dataset)):
                out, hidden = model.forward(dataset[t].unsqueeze(0), pasthidden)
                predictions.append([])
                rearranged.append([])
                errors.append([])
                hiddens.append(model.extract_hidden(hidden))
                if score_predictor is not None:
                    predicted_scores.append(score_predictor.predict(model.extract_hidden(hidden).numpy()))

                predictions[t].append(out.data.cpu()[0][0][channel_idx])
                pasthidden = model.repackage_hidden(hidden)
                for prediction_step in range(1, args.prediction_window_size):
                    out, hidden = model.forward(out, hidden)
                    predictions[t].append(out.data.cpu()[0][0][channel_idx])

                if t >= args.prediction_window_size:
                    for step in range(args.prediction_window_size):
                        rearranged[t].append(
                            predictions[step + t - args.prediction_window_size][args.prediction_window_size - 1 - step])
                    rearranged[t] =torch.FloatTensor(rearranged[t]).to(args.device).unsqueeze(0)
                    errors[t] = rearranged[t] - dataset[t][0][channel_idx]
                else:
                    rearranged[t] = torch.zeros(1,args.prediction_window_size).to(args.device)
                    errors[t] = torch.zeros(1, args.prediction_window_size).to(args.device)

        predicted_scores = np.array(predicted_scores)
        scores = []
        for error in errors:
            mult1 = error-mean.unsqueeze(0) # [ 1 * prediction_window_size ]
            mult2 = torch.inverse(cov) # [ prediction_window_size * prediction_window_size ]
            mult3 = mult1.t() # [ prediction_window_size * 1 ]
            score = torch.mm(mult1,torch.mm(mult2,mult3))
            scores.append(score[0][0])

        scores = torch.stack(scores)
        rearranged = torch.cat(rearranged,dim=0)
        errors = torch.cat(errors,dim=0)

        return scores, rearranged, errors, hiddens, predicted_scores
    
    def get_precision_recall(self, args, score, label, num_samples, beta=1.0, sampling='log', predicted_score=None):
        '''
        :param args:
        :param score: anomaly scores
        :param label: anomaly labels
        :param num_samples: the number of threshold samples
        :param beta:
        :param scale:
        :return:
        '''
        if predicted_score is not None:
            score = score - torch.FloatTensor(predicted_score).squeeze().to(args.device)

        maximum = score.max()
        if sampling=='log':
            # Sample thresholds logarithmically
            # The sampled thresholds are logarithmically spaced between: math:`10 ^ {start}` and: math:`10 ^ {end}`.
            th = torch.logspace(0, torch.log10(torch.tensor(maximum)), num_samples).to(args.device)
        else:
            # Sample thresholds equally
            # The sampled thresholds are equally spaced points between: attr:`start` and: attr:`end`
            th = torch.linspace(0, maximum, num_samples).to(args.device)

        precision = []
        recall = []

        for i in range(len(th)):
            anomaly = (score > th[i]).float()
            idx = anomaly * 2 + label
            tn = (idx == 0.0).sum().item()  # tn
            fn = (idx == 1.0).sum().item()  # fn
            fp = (idx == 2.0).sum().item()  # fp
            tp = (idx == 3.0).sum().item()  # tp

            p = tp / (tp + fp + 1e-7)
            r = tp / (tp + fn + 1e-7)

            if p != 0 and r != 0:
                precision.append(p)
                recall.append(r)

        precision = torch.FloatTensor(precision)
        recall = torch.FloatTensor(recall)


        f1 = (1 + beta ** 2) * (precision * recall).div(beta ** 2 * precision + recall + 1e-7)

        return precision, recall, f1


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
