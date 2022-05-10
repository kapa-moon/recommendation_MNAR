import torch


class LogisticRegressionModel(torch.nn.Module, ):
    def __init__(self,):
        super(LogisticRegressionModel, self).__init__()
        self.linear = torch.nn.Linear(7, 1)
    def forward(self, x):
        y_pred = torch.sigmoid(self.linear(x))
        return y_pred





