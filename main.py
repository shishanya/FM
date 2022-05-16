"""
代码仅供自己学习，无任何其他目的。
issue : 损失函数为nan，predict也为nan，可能是数据问题

solution : 下次对数据标准化，或者换个数据集试试
"""

from trainer import Train
from layer import FM
from utils import load_data
import warnings
warnings.filterwarnings("ignore")

config = {
    "latent_dim": 10,
    "lr": 0.001,
    "l2_regularization": 0.2,
    "use_cuda": False,
    "epoch": 100,
    "batch_size": 10,
    "device_id": 0,
    "num_features": 13,
    "bitch_size" : 10
}

if __name__ == '__main__':
    train_dataset, test_dataset = load_data()
    model = FM(config = config, p = config["num_features"])
    train = Train(model, config)
    train.train(train_dataset)
    train.evaluate(test_dataset)
