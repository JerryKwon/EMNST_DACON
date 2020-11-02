# evaluator.py
# trainning, evaluating CNN model

import torch
import numpy as np
from tqdm import tqdm

class Evaluator:
    def __init__(self, criterion, optimizer):
        self.criterion = criterion
        self.optimizer = optimizer

    def train(self, model, train_loader, valid_loader):
        DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

        if DEVICE == "cuda":
            model = model.to(DEVICE)

        trn_loss = 0.0
        trn_corrects = 0

        # itr_loader = iter(valid_loader)
        # data = next(itr_loader)
        # print(data)

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

    def predict(self, model, target_loader, output_type='test', val_dict=None):

        DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

        if DEVICE == "cuda":
            model = model.to(DEVICE)

        result_dict = dict()

        model.eval()
        with torch.no_grad():
            if output_type == 'valid':

                assert val_dict is not None

                ids = list(map(lambda x:x["img_id"],list(val_dict.values())))
                target_batch_size = target_loader.batch_size
                ids_per_batch = list(self.chunks(ids,target_batch_size))

                print("Valid Prediction Processing", "*" * 20)
                for batch_num, (inputs,labels) in enumerate(tqdm(target_loader)):
                    outputs = model(inputs)
                    preds = torch.argmax(outputs, 1)

                    batch_ids = np.array(ids_per_batch[batch_num])
                    preds = preds.data.numpy()

                    for idx, id in enumerate(batch_ids):
                        result_dict[id] = preds[idx]

            else:
                print("Test Prediction Processing","*"*20)
                for inputs,ids in tqdm(target_loader):

                    outputs = model(inputs)
                    preds = torch.argmax(outputs,1)

                    ids = ids.data.numpy()
                    preds = preds.data.numpy()

                    for idx, id in enumerate(ids):
                        result_dict[id] = preds[idx]

            return result_dict

    def chunks(self, lst, n):
        """Yield successive n-sized chunks from lst."""
        for i in range(0, len(lst), n):
            yield lst[i:i + n]