import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.metrics import confusion_matrix

class Calculate_score():
    def __init__(self, args):
        super(Calculate_score, self).__init__()
        self.seq_len = args.seq_len
        self.step_len = args.step_len

    def score(self, true_list, pred_list):
        f1 = f1_score(true_list, pred_list, average='macro')
        precision = precision_score(true_list, pred_list, average="macro")
        recall = recall_score(true_list, pred_list, average="macro")

        print(confusion_matrix(true_list, pred_list))
        print(f"f1 {f1} / precision {precision} / recall {recall}")

        return f1, precision, recall
    
    def back_score(self, true_list, pred_list):
        test_len = int(len(pred_list) / self.seq_len)
        
        # pred_list = np.array(pred_list)
        # true_list = np.array(true_list)
        pred_list_back = np.zeros(self.step_len*(test_len-1) + self.seq_len)
        true_list_back = np.zeros(self.step_len*(test_len-1) + self.seq_len)

        for i in range(test_len-1):
            start = self.seq_len * i
            back_i = self.step_len * i
            pred_list_back[back_i:back_i+self.step_len] = pred_list[start:(start+self.step_len)]
            true_list_back[back_i:back_i+self.step_len] = true_list[start:(start+self.step_len)]
        pred_list_back[-self.seq_len:] = pred_list[-self.seq_len:]
        true_list_back[-self.seq_len:] = true_list[-self.seq_len:]

        f1, precision, recall = self.score(true_list_back, pred_list_back)
        # f1 = f1_score(true_list, pred_list, average='macro')
        # precision = precision_score(true_list, pred_list, average="macro")
        # recall = recall_score(true_list, pred_list, average="macro")

        # print(confusion_matrix(true_list, pred_list))
        # print(f"f1 {f1} / precision {precision} / recall {recall}")

        return f1, precision, recall