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