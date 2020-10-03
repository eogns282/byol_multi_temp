import torch
import torch.nn as nn

class Distance_1D(nn.Module):
    r"""Implement of large margin cosine distance: :
    Args:
        in_features: size of each input sample
        out_features: size of each output sample
        s: norm of input feature
        m: margin
        cos(theta) - m
    """

    def __init__(self, out_features: int, num_classes: int):
        super(Distance_1D, self).__init__()
        self.out_features = out_features
        self.weight = nn.Parameter(torch.Tensor(num_classes, out_features), requires_grad=True)
        nn.init.xavier_uniform_(self.weight)

    def forward(self, input):
        output = torch.cdist(input, self.weight, 2)
        output = 2 * nn.Sigmoid()(output * -1)
        return output

# class Multitask(nn.Module):
#     r"""Implement of large margin cosine distance: :
#     Args:
#         in_features: size of each input sample
#         out_features: size of each output sample
#         s: norm of input feature
#         m: margin
#         cos(theta) - m
#     """
#
#     def __init__(self, backbone):
#         super(Multitask, self).__init__()
#         self.model = backbone
#         n_features = self.model.fc.in_features
#         self.model.fc = nn.Sequential()
#         self.fc1 = nn.Linear(n_features, 10, bias=True)
#         self.fc2 = Distance_1D(n_features,num_classes=10)
#
#     def forward(self, input):
#         output = self.model(input)
#         out_byol = self.fc1(output)
#         out_one = self.fc2(output)
#         return out_byol, out_one