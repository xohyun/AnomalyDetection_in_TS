import torch
import torch.nn as nn
import logging
import torch.optim as optim
from torch.autograd import Variable

from utils.utils import gpu_checking
from Trainer.base_trainer import base_trainer

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

    def train(self, n_epochs=2000):
        logging.debug('Starting training')
        cx_epoch_loss = list()
        cz_epoch_loss = list()
        encoder_epoch_loss = list()
        decoder_epoch_loss = list()

        for epoch in range(n_epochs):
            logging.debug('Epoch {}'.format(epoch))
            n_critics = 5

            cx_nc_loss = list()
            cz_nc_loss = list()

            for i in range(n_critics):
                cx_loss = list()
                cz_loss = list()

                for batch, sample in enumerate(self.data_loader):
                    loss = self.critic_x_iteration(sample)
                    cx_loss.append(loss)

                    loss = self.critic_z_iteration(sample)
                    cz_loss.append(loss)

                cx_nc_loss.append(torch.mean(torch.tensor(cx_loss)))
                cz_nc_loss.append(torch.mean(torch.tensor(cz_loss)))
                
            # print(f"{epoch}epoch loss : {torch.mean(torch.tensor(cx_nc_loss))}")
            
            logging.debug('Critic training done in epoch {}'.format(epoch))
            encoder_loss = list()
            decoder_loss = list()

            for batch, sample in enumerate(self.data_loader):
                enc_loss = self.encoder_iteration(sample)
                dec_loss = self.decoder_iteration(sample)
                encoder_loss.append(enc_loss)
                decoder_loss.append(dec_loss)

            cx_epoch_loss.append(torch.mean(torch.tensor(cx_nc_loss)))
            cz_epoch_loss.append(torch.mean(torch.tensor(cz_nc_loss)))
            encoder_epoch_loss.append(torch.mean(torch.tensor(encoder_loss)))
            decoder_epoch_loss.append(torch.mean(torch.tensor(decoder_loss)))
            logging.debug('Encoder decoder training done in epoch {}'.format(epoch))
            logging.debug('critic x loss {:.3f} critic z loss {:.3f} \nencoder loss {:.3f} decoder loss {:.3f}\n'.format(cx_epoch_loss[-1], cz_epoch_loss[-1], encoder_epoch_loss[-1], decoder_epoch_loss[-1]))

            if epoch % 10 == 0:
                torch.save(self.encoder.state_dict(), f'{self.args.save_path}_{self.args.model}_encoder')
                torch.save(self.decoder.state_dict(), f'{self.args.save_path}_{self.args.model}_decoder')
                torch.save(self.critic_x.state_dict(), f'{self.args.save_path}_{self.args.model}_critic_x')
                torch.save(self.critic_z.state_dict(), f'{self.args.save_path}_{self.args.model}_critic_z')
    
    def evaluation(self, test_loader):
        #---# prediction #---#
        preds = []
        critic_score = []
        for batch, sample in enumerate(test_loader):
            pred = self.decoder(self.encoder(sample))
            critic = self.critic_x(sample).detach().numpy() # [1,64,1]

            preds.append(pred.reshape(64,1000))
            critic_score.extend(critic.reshape(64))

        predictions = torch.cat(preds)
        return predictions

    def critic_x_iteration(self, sample):
        self.optim_cx.zero_grad()
        
        # x = sample['signal'].view(1, batch_size, signal_shape)
        x = sample.view(1, self.batch_size, self.features)
        valid_x = self.critic_x(x)
        valid_x = torch.squeeze(valid_x)
        critic_score_valid_x = torch.mean(torch.ones(valid_x.shape) * valid_x) # Wasserstein Loss

        #The sampled z are the anomalous points - points deviating from actual distribution of z (obtained through encoding x)
        z = torch.empty(1, self.batch_size, self.latent_space_dim).uniform_(0, 1)
        x_ = self.decoder(z)
        fake_x = self.critic_x(x_)
        fake_x = torch.squeeze(fake_x)
        critic_score_fake_x = torch.mean(torch.ones(fake_x.shape) * fake_x)  #Wasserstein Loss

        alpha = torch.rand(x.shape)
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
        x = sample.view(1, self.batch_size, self.features)
        z = self.encoder(x)
        valid_z = self.critic_z(z)
        valid_z = torch.squeeze(valid_z)
        critic_score_valid_z = torch.mean(torch.ones(valid_z.shape) * valid_z)

        z_ = torch.empty(1, self.batch_size, self.latent_space_dim).uniform_(0, 1)
        fake_z = self.critic_z(z_)
        fake_z = torch.squeeze(fake_z)
        critic_score_fake_z = torch.mean(torch.ones(fake_z.shape) * fake_z) #Wasserstein Loss

        wl = critic_score_fake_z - critic_score_valid_z

        alpha = torch.rand(z.shape)
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
        # x = sample['signal'].view(1, batch_size, signal_shape)
        x = sample.view(1, self.batch_size, self.features)
        valid_x = self.critic_x(x)
        valid_x = torch.squeeze(valid_x)
        critic_score_valid_x = torch.mean(torch.ones(valid_x.shape) * valid_x) #Wasserstein Loss

        z = torch.empty(1, self.batch_size, self.latent_space_dim).uniform_(0, 1)
        x_ = self.decoder(z)
        fake_x = self.critic_x(x_)
        fake_x = torch.squeeze(fake_x)
        critic_score_fake_x = torch.mean(torch.ones(fake_x.shape) * fake_x)

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
        x = sample.view(1, self.batch_size, self.features)
        z = self.encoder(x)
        valid_z = self.critic_z(z)
        valid_z = torch.squeeze(valid_z)
        critic_score_valid_z = torch.mean(torch.ones(valid_z.shape) * valid_z)

        z_ = torch.empty(1, self.batch_size, self.latent_space_dim).uniform_(0, 1)
        fake_z = self.critic_z(z_)
        fake_z = torch.squeeze(fake_z)
        critic_score_fake_z = torch.mean(torch.ones(fake_z.shape) * fake_z)

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

