import argparse

class Args:
    def __init__(self):
        self.args = self.parse_arguments()

    def parse_arguments(self):
        parser = argparse.ArgumentParser()
        
        #---# Mode #---#
        parser.add_argument("--mode", default="all", choices=["train", "test", "all"])
        parser.add_argument("--seed", default=1004, type=int)
        parser.add_argument("--valid_setting", default=False, choices=[True, False])

        #---# device #---#
        parser.add_argument("--device", default=0, help="cpu or gpu number")

        #---# Path #---#
        parser.add_argument("--data_path", default="./TimeSeries-Anomaly-Detection-Dataset/data/")
        parser.add_argument("--save_path", default="./model_save/")

        #---# Dataset #---#
        nab_list = ['ec2'] ############# test
        parser.add_argument("--dataset", default='NAB', choices=['MSL', 'NAB', 'SMAP', 'SMD', 'WADI'])
        parser.add_argument("--choice_data", default=None)
        
        
        #---# Dataset setting #---#
        parser.add_argument("--seq_len", default=10)
        parser.add_argument("--pred_len", default=1)
        parser.add_argument("--step_len", default=1)

        #---# Model #---#
        parser.add_argument("--model", type=str, default="AE", choices=["AE"])

        parser.add_argument("--lr", type=float, default=0.001)
        parser.add_argument("--wd", type=float, default=0.0001)
        parser.add_argument("--batch_size", type=int, default=16)
        parser.add_argument("--epoch", type=int, default=2)

        parser.add_argument('--scheduler', '-sch')
        if parser.parse_known_args()[0].scheduler == 'exp':
            parser.add_argument('--gamma', type=float, required=True)
        elif parser.parse_known_args()[0].scheduler == 'step':
            parser.add_argument('--step_size', type=int, required=True, default=10)
            parser.add_argument('--gamma', type=float, required=True, default=0.5)
        elif parser.parse_known_args()[0].scheduler == 'multi_step':
            parser.add_argument('--milestones', required=True) # type=str2list_int
            parser.add_argument('--gamma', type=float, required=True)
        elif parser.parse_known_args()[0].scheduler == 'plateau':
            parser.add_argument('--factor', type=float, required=True)
            parser.add_argument('--patience', type=int, required=True)
        elif parser.parse_known_args()[0].scheduler == 'cosine':
            parser.add_argument('--T_max', type=float, help='Max iteration number', default=50)
            parser.add_argument('--eta_min', type=float, help='minimum learning rate', default=0)
        elif parser.parse_known_args()[0].scheduler == 'one_cycle': #default로 cosine이라 anneal_strategy를 따로 변수로 안잡음
            parser.add_argument('--max_lr', type=float, default=0.1)
            parser.add_argument('--steps_per_epoch', type=int, default=10) # 증가하는 cycle의 반
            parser.add_argument('--cycle_epochs', type=int, default=10)

        parser.add_argument("--criterion", type=str, default="MSE") #triplet
        parser.add_argument("--optimizer", type=str, default="SGD")   # AdamW, SGD

        args = parser.parse_args()
        return args

