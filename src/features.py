# %%
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

def main():
    # テストデータ
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                        download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=4,
                                            shuffle=False, num_workers=2)
    # データ
    data=testloader
    # CNN
    cnn=AlexNet(mode=True, model_path='./model/parameter.pth')
    # 特徴量抽出
    features=cnn.features(data)
    np.savetxt('./features.txt', features)

if __name__=='__main__':
    main()
# %%
