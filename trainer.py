import torch


class Train():
    def __init__(self, model, config):
        self._model = model
        self._config = config
        self._optimizer = torch.optim.SGD(self._model.parameters(), lr=config['lr'],
                                          weight_decay=config['l2_regularization'])
        self._loss_func = torch.nn.MSELoss()

    def _train_single_batch(self, x, labels):
        """
        对单个小批量数据进行训练
        """
        self._optimizer.zero_grad()
        y_predict = self._model(x)
        print(y_predict)
        loss = self._loss_func(y_predict.view(-1, 1), labels)

        loss.backward()
        self._optimizer.step()

        loss = loss.item()  # The item() method extracts the loss’s value as a Python float.
        return loss, y_predict

    def _train_an_epoch(self, train_loader, epoch_id):
        """
        训练一个Epoch，即将训练集中的所有样本全部都过一遍
        """
        self._model.train()
        total = 0
        X = train_loader[:,:-1]
        Labels = train_loader[:,-1]
        length = len(X)
        print(X.shape, Labels.shape)
        for index in range((len(X) // self._config["bitch_size"]) + 1):
            end = min(self._config["bitch_size"], length - self._config["bitch_size"] * index)
            x, labels = X[self._config["bitch_size"] * index : self._config["bitch_size"] * index + end], Labels[self._config["bitch_size"] * index : self._config["bitch_size"] * index + end]
            x = torch.FloatTensor(x)
            labels = torch.FloatTensor(labels)
            if self._config["use_cuda"] is True:
                x, labels = x.cuda(), labels.cuda()
            loss, y_predict = self._train_single_batch(x, labels)
            total += loss
        print("Training Epoch: %d, total loss: %f" % (epoch_id, total))

    def train(self, train_dataset):
        self.use_cuda()
        for epoch in range(self._config["epoch"]):
            print('-' * 20 + ' Epoch {} starts '.format(epoch) + '-' * 20)
            self._train_an_epoch(train_dataset, epoch_id=epoch + 1)

    def evaluate(self, test_dataset):
        x = test_dataset[:, :-1]
        labels = test_dataset[:, -1]

        x, labels = torch.FloatTensor(x), torch.FloatTensor(labels)
        if self._config["use_cuda"] is True:
            x = x.cuda()
        y_predict = self._model(x)

    def use_cuda(self):
        if self._config['use_cuda'] is True:
            assert torch.cuda.is_available(), 'CUDA is not available'
            torch.cuda.set_device(self._config['device_id'])
            self._model.cuda()

    def save(self):
        pass
