import os
import random
import torch
import torch.backends.cudnn as cudnn
import numpy as np
def create_folder(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print ('Error: Creating directory. ' +  directory)
 
def gpu_checking(args) :
    device = torch.device(f'cuda:{str(args.device)}' if torch.cuda.is_available() else 'cpu')
    # os.environ["CUDA_VISIBLE_DEVICES"] = '0, 1, 2, 3'
    # os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device)
    return device

def fix_random_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    os.environ['PYTHONHASHSEED'] = str(args.seed)

    torch.manual_seed(args.seed)
    # Multi-GPU
    if args.device == "multi":
        torch.cuda.manual_seed_all(args.seed)
    # Single-GPU
    else:
        torch.cuda.manual_seed(args.seed)
    cudnn.benchmark = True  # If you want to set randomness, cudnn.benchmark = False
    cudnn.deterministic = True  # If you want to set randomness, cudnn.benchmark = True
    print(f"[Control randomness]\nseed: {args.seed}")

def find_bundle(queue):
    packet = []
    tmp = []
    v = queue.pop(0)
    tmp.append(v)
    
    while(len(queue)>0):
        vv = queue.pop(0)
        if v+1 == vv:
            tmp.append(vv)
            v = vv
        else:
            packet.append(tmp)
            tmp = []
            tmp.append(vv)
            v = vv
        
    packet.append(tmp)
    return packet

def save_checkpoint(self, epoch):
    create_folder(os.path.join(self.args.save_path, "checkpoints"))
    torch.save({
        'epoch': epoch,
        'model_state_dict': self.model.state_dict(),
        'optimizer_state_dict': self.optimizer.state_dict(),
        # 'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None
    }, os.path.join(self.args.save_path, f"checkpoints/{epoch}.tar"))

def load_module_func(module_name):
    '''
    Import different files depending on the module name
    '''
    import importlib
    mod = importlib.import_module(module_name)
    return mod

def make_csv(df, args):
    '''
    make csv file
    df : dataframe
    args : arguments
    '''
    df.to_csv(f'{args.csv_path}{args.model}_{args.dataset}_{args.choice_data}_lr{args.lr}_wd{args.wd}_seq{args.seq_len}_step{args.step_len}_batch{args.batch_size}_epoch{args.epoch}_score{args.score}_calc{args.calc}_num_stack_{args.stack_num}_ratio_{args.recon_ratio}.csv', header = True, index = False)
    # if args.dataset == 'NAB':
    #     df.to_csv(f'{args.csv_path}{args.model}_{args.dataset}_{args.choice_data}_lr{args.lr}_wd{args.wd}_seq{args.seq_len}_step{args.step_len}_batch{args.batch_size}_epoch{args.epoch}_score{args.score}_calc{args.calc}.csv', header = True, index = False)
    # else:
    #     df.to_csv(f'{args.csv_path}{args.model}_{args.dataset}_lr{args.lr}_wd{args.wd}_seq{args.seq_len}_step{args.step_len}_batch{args.batch_size}_epoch{args.epoch}_score{args.score}_calc{args.calc}.csv', 
    #                 header = True, index = False)

#---# get number of parameter function1 #---#
def print_arch(model, model_name='model'):
    print(f"| {model_name} Arch: ", model)
    num_params(model, model_name=model_name)

#---# get number of parameter function2 #---#
def num_params(model, print_out=True, model_name="model"):
    parameters = filter(lambda p: p.requires_grad, model.parameters())
    parameters = sum([np.prod(p.size()) for p in parameters]) / 1_000_000 # 자리수구분위한
    if print_out:
        print(f'| {model_name} Trainable Parameters: %.3fM' % parameters)
    return parameters