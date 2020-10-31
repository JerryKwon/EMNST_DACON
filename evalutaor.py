# evaluator.py
# trainning, evaluating CNN model

import torch

class Evaluator:
    def __init__(self, model, criterion, optimizer):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer

    def train(self, train_loader, valid_loader):
        DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
        if DEVICE == "cuda":
            model = self.model.to(DEVICE)

        trn_loss = 0.0
        trn_corrects = 0

        model.train()
        for batch_num, (inputs, labels) in enumerate(train_loader):
            model.zero_grad()
            if DEVICE == "cuda":
                inputs.to(DEVICE)
                labels.to(DEVICE)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            loss = self.criterion(outputs, labels)
            running_corrects = torch.sum(preds == labels.data).item()

            loss.backward()
            self.optimizer.step()

            trn_loss += loss.item() * inputs.size(0)
            trn_corrects += running_corrects

            del loss
            del outputs

        trn_loss = trn_loss / len(train_loader.dataset)
        trn_acc = trn_corrects / len(train_loader.dataset)

        model.eval()
        with torch.no_grad():
            val_loss = 0.0
            val_corrects = 0
            for batch_num, (inputs, labels) in enumerate(valid_loader):
                if DEVICE == "cuda":
                    inputs = inputs.to(DEVICE)
                    labels = labels.to(DEVICE)

                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)

                loss = self.criterion(outputs, labels)
                running_corrects = torch.sum(preds == labels.data).item()

                val_loss += loss * inputs.size(0)
                val_corrects += running_corrects

                del loss
                del outputs

            val_loss = val_loss / len(valid_loader.dataset)
            val_acc = val_corrects / len(valid_loader.dataset)

        return trn_loss, val_loss, trn_acc, val_acc

    def predict(self, model, test_loader):
        result_dict = dict()

        model.eval()
        with torch.no_grad():

            for inputs,ids in test_loader:

                outputs = model(inputs)
                preds = torch.argmax(outputs,1)

                ids = ids.data.numpy()
                preds = preds.data.numpy()

                for idx, id in enumerate(ids):
                    result_dict[id] = preds[idx]

        return result_dict
