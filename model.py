# 以下を「model.py」に書き込み
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image

classes_ja = ['りんご', '観賞魚', '赤ちゃん', '熊', 'ビーバー', 'ベッド', '蜂', 'カブトムシ',
                        '自転車', 'ボトル', 'ボウル', '男の子', '橋', 'バス', '蝶', 'ラクダ',
                        '缶', '城', '毛虫', '牛', '椅子', 'チンパンジー', '時計',
                        '雲', 'ゴキブリ', 'ソファ', 'カニ', 'ワニ', 'カップ', '恐竜',
                        'イルカ', 'ゾウ', 'ヒラメ', '森', 'キツネ', '女の子', 'ハムスター',
                        '馬', 'カンガルー', 'キーボード', 'ランプ', '芝刈り機', 'ヒョウ', 'ライオン',
                        'トカゲ', 'ロブスター', '男性', 'もみじ', 'オートバイ', '山', 'ねずみ',
                        'キノコ', 'オーク', 'オレンジ', '蘭', 'カワウソ', 'ヤシ', '洋ナシ',
                        'ピックアップトラック', '松', '平野', '皿', 'ポピー', 'ヤマアラシ',
                        'ふくろネズミ', 'ウサギ', 'アライグマ', 'エイ', '道路', 'ロケット', 'バラ',
                        '海', 'アザラシ', 'サメ', 'トガリネズミ', 'スカンク', '超高層ビル', 'カタツムリ', 'ヘビ',
                        'クモ', 'リス', '路面電車', 'ひまわり', 'りんご', 'テーブル',
                        'タンク', '携帯電話', 'テレビ', '引き金', 'トラクター', '電車', '',
                        'チューリップ', 'カメ', 'ワードロープ', 'クジラ', '柳', '音楽', '女の子',
                        'ミミズ']
classes_en = ['apple', 'aquarium_fish', 'baby', 'bear', 'beaver', 'bed', 'bee', 'beetle',
                        'bicycle', 'bottle', 'bowl', 'boy', 'bridge', 'bus', 'butterfly', 'camel',
                        'can', 'castle', 'caterpillar', 'cattle', 'chair', 'chimpanzee', 'clock',
                        'cloud', 'cockroach', 'couch', 'crab', 'crocodile', 'cup', 'dinosaur',
                        'dolphin', 'elephant', 'flatfish', 'forest', 'fox', 'girl', 'hamster',
                        'house', 'kangaroo', 'keyboard', 'lamp', 'lawn_mower', 'leopard', 'lion',
                        'lizard', 'lobster', 'man', 'maple_tree', 'motorcycle', 'mountain', 'mouse',
                        'mushroom', 'oak_tree', 'orange', 'orchid', 'otter', 'palm_tree', 'pear',
                        'pickup_truck', 'pine_tree', 'plain', 'plate', 'poppy', 'porcupine',
                        'possum', 'rabbit', 'raccoon', 'ray', 'road', 'rocket', 'rose',
                        'sea', 'seal', 'shark', 'shrew', 'skunk', 'skyscraper', 'snail', 'snake',
                        'spider', 'squirrel', 'streetcar', 'sunflower', 'sweet_pepper', 'table',
                        'tank', 'telephone', 'television', 'tiger', 'tractor', 'train', 'trout',
                        'tulip', 'turtle', 'wardrobe', 'whale', 'willow_tree', 'wolf', 'woman',
                        'worm']
n_class = len(classes_ja)
img_size = 32

# CNNのモデル
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16*5*5, 256)
        self.dropout = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(256, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16*5*5)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

def predict(img):
    # モデルへの入力
    img = img.convert("RGB")
    img = img.resize((img_size, img_size))
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.0, 0.0, 0.0), (1.0, 1.0, 1.0))  # 平均値を0、標準偏差を1に
                                ])
    img = transform(img)
    x = img.reshape(1, 3, img_size, img_size)

    # 訓練済みモデル
    net = Net()
    net.load_state_dict(torch.load(
        "model_cnn.pth", map_location=torch.device("cpu")
        ))
    
    # 予測
    net.eval()
    y = net(x)

    # 結果を返す
    y_prob = torch.nn.functional.softmax(torch.squeeze(y))  # 確率で表す
    sorted_prob, sorted_indices = torch.sort(y_prob, descending=True)  # 降順にソート
    return [(classes_ja[idx], classes_en[idx], prob.item()) for idx, prob in zip(sorted_indices, sorted_prob)]
