import torch.nn as nn

"""Model"""
class Model(nn.Module):
    """コンストラクタ"""
    def __init__(self):
        super(Model, self).__init__()
        # 特徴量抽出
        self.__features=nn.Sequential(
            nn.Conv2d(3, 32, 3),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3),
            nn.MaxPool2d((2,2))
        )
        # 1次元データに変換
        self.__flatten=nn.Sequential(
            # 1次元データに変換
            nn.Flatten()
        )
        # 分類器
        self.__classifier=nn.Sequential(
            nn.Linear(6272, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 10)
        )
        
    """順伝播"""
    def forward(self, x):
        # 特徴量抽出
        x=self.__features(x)
        # 1次元データに変換
        x=self.__flatten(x)
        # 分類器
        y=self.__classifier(x)
        return y
    
    """特徴量"""
    def features(self, x):
        x=self.__features(x)
        return x