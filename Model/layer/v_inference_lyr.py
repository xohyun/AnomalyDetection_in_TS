import torch.nn as nn


class v_inference_lyr(nn.Module):
    def __init__(self, seq_len, feature_num):
        super().__init__()
        self.seq_len = seq_len
        self.feature_num = feature_num

        self.n = int(self.seq_len * 0.8) * self.feature_num
        self.hidden1 = int(self.n / 4)
        self.hidden2 = int(self.n / 8)

        self.fc = nn.Sequential(
            nn.Linear(self.n, self.feature_num)
        )

    def forward(self, x):
        batch = x.shape[0]
        x = x.reshape(batch, -1)
        return self.fc(x)
