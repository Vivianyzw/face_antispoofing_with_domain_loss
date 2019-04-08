import torch
import torch.nn as nn


class MMD_loss(nn.Module):
    def __init__(self):
        super(MMD_loss, self).__init__()

    def guassian_kernel(self, source, target):
        total = torch.cat([source, target], dim=0)

        total0 = total.unsqueeze(0).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)), int(total.size(2)), int(total.size(3)))
        total1 = total.unsqueeze(1).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)), int(total.size(2)), int(total.size(3)))
        L2_distance = ((total0-total1)**2).sum(2)

        bandwidth_list = [2, 5, 10, 20, 40, 80]
        kernel_val = [torch.exp(-L2_distance / bandwidth_temp) for bandwidth_temp in bandwidth_list]
        return sum(kernel_val)

    def forward(self, source, target):
        batch_size = int(source.size()[0])
        kernels = self.guassian_kernel(source, target)
        XX = kernels[:batch_size, :batch_size]
        YY = kernels[batch_size:, batch_size:]
        XY = kernels[:batch_size, batch_size:]
        YX = kernels[batch_size:, :batch_size]
        loss = torch.mean(XX + YY - XY -YX)
        return loss


if __name__ == '__main__':
    import numpy as np
    a = np.array([1, 2, 3, 4])
    a = torch.from_numpy(a)
    print(a)
