from alexnet import AlexNet

import torch
import torchvision
import torchvision.transforms as transforms

import random
import numpy as np

# random seedを設定
seed = 2023
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

# 実行文
def main():
    # 訓練データ
    transform = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                            shuffle=True, num_workers=2)
    print ('train_dataset = ', len(trainset))
    
    # データ
    data=trainloader

    # CNN
    cnn=AlexNet()

    # 学習
    cnn.update(data, mode=True, epoch = 300)

if __name__=='__main__':
    main()

