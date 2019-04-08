import torch
from model_torch import *
import torch.utils.data as dataLoader
import os
from mmd_loss_torch import *
from image_loader import *
import tensorflow as tf
import numpy as np


def single_gpu_train():
    EPOCH = 10000
    BATCH_SIZE = 8
    ALPHA = 0.5
    IMAGE_SIZE = 128
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device_ids = [0]

    # net = MMDNet()
    net = MobileNet()

    net = net.to(device)

    dataset = mytraindata(r"/path/to/real_source",
                          r"/path/to/real_target",
                          r"/path/to/fake_source",
                          r"/path/to/fake_target",
                          True, IMAGE_SIZE)

    data_loader = dataLoader.DataLoader(dataset, batch_size=BATCH_SIZE)

    optimizer = torch.optim.Adam([{'params': net.parameters()}], lr=0.001)
    domain_criterion = MMD_loss()
    classification_criterion = torch.nn.CrossEntropyLoss()

    for epoch in range(EPOCH):
        running_loss = 0.
        running_acc = 0.
        for i, data in enumerate(data_loader):
            image, label = data # image 224*224*(3*4)
            image = image.to(device)
            label = label.to(device)

            # 此处可以增加预处理
            real_source_feature, real_source_pred = net(image[:, :3, :, :])
            real_target_feature, real_target_pred = net(image[:, 3:6, :, :])
            fake_source_feature, fake_source_pred = net(image[:, 6:9, :, :])
            fake_target_feature, fake_target_pred = net(image[:, 9:12, :, :])

            real_loss = domain_criterion(real_source_feature, real_target_feature)
            fake_loss = domain_criterion(fake_source_feature, fake_target_feature)
            real_cls_loss = classification_criterion(real_source_pred, label[:,0].long())
            fake_cls_loss = classification_criterion(fake_source_pred, label[:,2].long())

            domain_loss = real_loss + fake_loss
            classification_loss = real_cls_loss + fake_cls_loss

            loss = classification_loss + ALPHA * domain_loss
            print('Epoch: %d | iter: %d | loss: %.10f | domain loss: %.10f | classification loss: %.10f' % (
            epoch, i, float(loss), float(domain_loss), float(classification_loss)))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # running_loss += loss.data[0]
            # _, predict = torch.max(output, 1)
            # correct_num = (predict == label).sum()
            # running_acc += correct_num.data[0]

        # if epoch % 100 == 99:
        model_name = os.path.join('models/model_%d.pkl' % epoch)
        torch.save(net.state_dict(), model_name)


if __name__ == '__main__':
    single_gpu_train()
    # multi_gpu_train()








