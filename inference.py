# inference.py
# training cnn model and evaluate to test dataset and return result.

import os
import argparse
import warnings
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch import optim
from torchvision import transforms

from dataset_loader import DatasetLoader
from emnst_dataset import EMNST_Dataset
from model import CustomCNN
from evalutaor import Evaluator

from tqdm import tqdm

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def output_type(v):
    if v.lower() in ('valid','test'):
        return v.lower()
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def main():

    warnings.filterwarnings("ignore")

    parser = argparse.ArgumentParser(description="excute inference.py")
    parser.add_argument('--is_generate', type=str2bool,
                        help='Whether Generating Splited Dataset using train / test.csv')
    parser.add_arguments('--n_folds', type=int, help='Number of Split folds on train dataset')
    # parser.add_argument('--output', type=output_type, help='Dataset what getting prediction')
    parser.add_argument('--epochs', type=int, help='EPOCH size for DL')
    parser.add_argument('--batch_size', type=int, help='BATCH size for DL')
    parser.add_argument('--predict_file', type=str, help='filename for final prediction')


    args = parser.parse.args()

    is_generate = args.is_generate
    n_folds = args.n_folds
    # output = args.output
    EPOCHS = args.epochs
    BATCH_SIZE = args.batch_size
    RESULT_FILENAME = args.predict_file

    loader = DatasetLoader()

    if not is_generate:
        loader.split(n_folds)

    trn_dict, val_dict, test_dict = loader.load_split(n_folds)

    train_transforms = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomRotation(30),
        transforms.RandomHorizontalFlip(),
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])

    valid_transforms = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])

    train_dataset = EMNST_Dataset(img_dict=trn_dict, img_width=28, img_height=28, transform=train_transforms)
    valid_dataset = EMNST_Dataset(img_dict=val_dict, img_width=28, img_height=28, transform=valid_transforms)
    test_dataset = EMNST_Dataset(img_dict=test_dict, img_width=28, img_height=28, transform=valid_transforms)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Available Device is {DEVICE}")

    model = CustomCNN()
    criterion = nn.CrossEntropyLoss()
    learning_rate = 0.001
    optimizer = optim(model.parameters(),learning_rate)

    evaluator = Evaluator(model, criterion, optimizer)

    best_loss = np.inf()
    model_path = os.path.join(loader.get_basepath(),'output','models')

    if not os.path.isdir(model_path):
        os.mkdir(model_path)

    model_state_file = os.path.join(model_path,'EMNST_CNN.pt')

    if os.path.isfile(model_state_file):
        model.load_state_dict(torch.load(model_state_file))

    for epoch in range(EPOCHS):
        train_loss, valid_loss, train_acc, valid_acc = evaluator.train(model,train_loader, valid_loader, optimizer, criterion)
        if valid_loss < best_loss:
            torch.save(model.state_dict(), model_state_file)
            best_loss = valid_loss
        print("epoch: {}/{} | trn loss: {:.4f} | val loss: {:.4f} | trn acc: {:.4f} | val acc: {:.4f}".format(
            epoch + 1, EPOCHS, train_loss, valid_loss,train_acc, valid_acc
        ))

    target_loader = valid_loader if output == "valid" else test_loader

    result_dict = evaluator.predict(model, target_loader,)

    loader.submit(result_dict,RESULT_FILENAME)
