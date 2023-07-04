import torch
from tqdm import tqdm
import os
import numpy as np
import torch.nn  as nn
from .model import Model


"""AlexNet"""
class AlexNet(object):
    """コンストラクタ"""
    def __init__(self, mode=False, model_path=''):
        # デバイス設定 GPU or CPU
        self.__device="cuda" if torch.cuda.is_available() else "cpu"
        # モデル定義
        self.__model=Model().to(self.__device)

        if mode:
            # 学習済みモデル読み込み
            self.__model.load_state_dict(torch.load(model_path))
            self.__model.eval()

        # 学習係数
        self.__lr=1e-3
        # 損失関数:交差エントロピー
        self.__loss_func=nn.CrossEntropyLoss()
        # 最適化アルゴリズム:SGD
        self.__opt=torch.optim.SGD(self.__model.parameters(), lr=self.__lr)

        # save file path
        self.FILE_PATH=os.path.join('./model')

        # フォルダを生成
        if not os.path.exists(self.FILE_PATH):
            os.mkdir(self.FILE_PATH)

        # 損失値格納用変数
        self.__loss_history=[]
        
    """update:学習"""
    def update(self, data, mode=False, epoch=100):
        # epoch=tqdm(epoch)
        for e in range(epoch):
            sum_log=0
            # パラメータ計算
            for batch, (X, y) in enumerate(data):
                # 28*28を784次元に変換
                # X=X.reshape(784)
                # device調整
                X=X.to(self.__device)
                y=y.to(self.__device)
                # 学習用データXをAutoEncoderモデルに入力 -> 計算結果 出力Y
                pred_y=self.__model(X)

                # 損失計算(ラベルYと予測Yとの交差エントロピーによる損失計算)
                loss=self.__loss_func(pred_y, y)

                # 誤差逆伝播を計算
                # 勾配値を0にする
                self.__opt.zero_grad()
                # 逆伝播を計算
                loss.backward()
                # 勾配を計算
                self.__opt.step()
                
                loss=loss.item()
                sum_log+=loss
            # 損失を格納
            self.__loss_history.append(sum_log)
            print(f'epoch:{e}, loss:{sum_log}')
            if(sum_log < 5): #損失が5を下回ったら終了
                break

        # 損失保存
        if mode:
            """汎用的な保存方法を検討中"""
            # ファイル path
            LOSS_SAVE=os.path.join(self.FILE_PATH, 'loss.txt')
            # 損失結果 保存
            np.savetxt(LOSS_SAVE, self.__loss_history)
            # パラメータ保存
            PARAM_SAVE=os.path.join(self.FILE_PATH, 'parameter.pth')
            # 学習したパラメータを保存
            torch.save(self.__model.state_dict(), PARAM_SAVE)
        
    """test_accuracy:テストデータを使った精度評価"""
    def test_accuracy(self, data, mode=False):
        data=tqdm(data)
        # 勾配なし
        with torch.no_grad():
            # 汎用的なデータセットに対応
            n=0
            # 精度
            acc=0
            # 精度
            correct=0
            # ラベル数の合計値
            total=0
            # パラメータ計算
            for batch, (X, y) in enumerate(data):
                # device調整
                X=X.to(self.__device)
                y=y.to(self.__device)
                # 予測
                pred=self.__model(X)
                # 精度計算
                correct+=(pred.argmax(dim=1) == y).type(torch.float).sum().item()
                # 合計
                total+=y.size(0)
                # データ数 計算
                n+=1
            
            # 精度[%]
            acc=100*(correct/total)
        
        print("\n ====================== \n")
        print(f"acc:{acc}")
        print("\n ====================== \n")

                # 損失保存
        if mode:
            """汎用的な保存方法を検討中"""
            # パラメータ保存
            PARAM_SAVE=os.path.join(self.FILE_PATH, 'acc.txt')
            # 学習したパラメータを保存
            np.savetxt(PARAM_SAVE, [acc])

        return acc
    
    """prediction:予測"""
    def prediction(self, X):
        X=X.to(self.__device)
        # 予測
        pred=self.__model(X)
        
        print("\n ====================== \n")
        print(f"y:{pred}")
        print("\n ====================== \n")     

        return pred
    
    """features:特徴量抽出"""
    def features(self, data):
        x, y=data
        x=x.to(self.__device)
        x=self.__model.features(x)
        return x
