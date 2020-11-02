# train.py
# training cnn model and return trained model.


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
from emnst_dataset import EMNST_Dataset, EMNST_Test_Dataset
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
    parser.add_argument('--is_generated', type=str2bool,
                        help='Whether Generating Splited Dataset using train / test.csv')
    parser.add_argument('--n_folds', type=int, help='Number of Split folds on train dataset')
    parser.add_argument('--epochs', type=int, help='EPOCH size for DL')
    parser.add_argument('--batch_size', type=int, help='BATCH size for DL')
    parser.add_argument('--get_pretrained', type=str, help='return pretrained weight model')
    parser.add_argument('--use_pretrained', type=str, help='using pretrained model for training')
    parser.add_argument('--model_file', type=str, help='filename for trained model')


    args = parser.parse_args()

    is_generated = args.is_generated
    n_folds = args.n_folds
    EPOCHS = args.epochs
    BATCH_SIZE = args.batch_size
    GET_PRETRAINED = args.get_pretrained
    USE_PRETRAINED = args.use_pretrained
    RESULT_FILENAME = args.model_file

    loader = DatasetLoader()

    if not is_generated:
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
    test_dataset = EMNST_Test_Dataset(img_dict=test_dict, img_width=28, img_height=28, transform=valid_transforms)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Available Device is {DEVICE}")

    model = CustomCNN()
    criterion = nn.CrossEntropyLoss()
    learning_rate = 0.001
    optimizer = optim.Adam(model.parameters(),lr=learning_rate)

    evaluator = Evaluator(criterion, optimizer)

    model_path = os.path.join(loader.get_basepath(),'output','models')

    if not os.path.isdir(model_path):
        os.mkdir(model_path)

    model_state_file = os.path.join(model_path,'EMNST_CNN.pt')
    model_val_loss_file = os.path.join(model_path,'EMNST_CNN_SCORE.txt')

    if GET_PRETRAINED:
        model.load_state_dict(torch.load(model_state_file))
        torch.save(os.path.join(model_path,RESULT_FILENAME))

    else:
        if USE_PRETRAINED:
            if os.path.isfile(model_state_file):
                model.load_state_dict(torch.load(model_state_file))
                with open(model_val_loss_file, 'r') as f:
                    best_loss_init = float(f.read())
                    best_loss = best_loss_init

        else:
            best_loss_init = np.inf
            best_loss = best_loss_init

        for epoch in range(EPOCHS):
            train_loss, valid_loss, train_acc, valid_acc = evaluator.train(model, train_loader, valid_loader)
            if valid_loss < best_loss:
                print("valid_loss broken from {:.4f} to {:.4f}".format(best_loss,valid_loss))
                torch.save(model.state_dict(), model_state_file)
                best_loss = valid_loss
            print("epoch: {}/{} | trn loss: {:.4f} | val loss: {:.4f} | trn acc: {:.4f} | val acc: {:.4f}".format(
                epoch + 1, EPOCHS, train_loss, valid_loss,train_acc, valid_acc
            ))

        if best_loss != best_loss_init:
            model.load_state_dict(torch.load(model_state_file))
            with open(model_val_loss_file, 'w') as f:
                f.write(str(best_loss.item()))

        else:
            model.load_state_dict(torch.load(model_state_file))

        torch.save(os.path.join(model_path,RESULT_FILENAME))

if __name__ == '__main__':
    main()