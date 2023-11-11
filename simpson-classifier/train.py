import wandb
from enum import Enum
import torch
import torch.nn as nn
from tqdm import tqdm
import uuid
import argparse

from model import MyCnn
from datasets import get_datasets


class rightOptimizers(Enum):
    Adagrad = 0
    AdamW = 1
    SGD = 2
    RMSprop = 3


class rightLosses(Enum):
    CrossEntropy= 0
    NLLLoss = 1
    KLDivLoss = 2
    

def fit_epoch(mod, train_data, criterion, optimizer):
    running_loss = 0.0
    running_corrects = 0
    processed_data = 0
  
    for inputs, labels in train_data:
        inputs = inputs.to(DEVICE)
        labels = labels.to(DEVICE)
        optimizer.zero_grad()

        outputs = mod(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        preds = torch.argmax(outputs, 1)
        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)
        processed_data += inputs.size(0)
              
    train_loss = running_loss / processed_data
    train_acc = running_corrects.cpu().numpy() / processed_data
    return train_loss, train_acc


def eval_epoch(mod, val_data, criterion):
    mod.eval()
    running_loss = 0.0
    running_corrects = 0
    processed_size = 0

    for inputs, labels in val_data:
        inputs = inputs.to(DEVICE)
        labels = labels.to(DEVICE)

        with torch.set_grad_enabled(False):
            outputs = mod(inputs)
            loss = criterion(outputs, labels)
            preds = torch.argmax(outputs, 1)

        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)
        processed_size += inputs.size(0)
    val_loss = running_loss / processed_size
    val_acc = running_corrects.double() / processed_size
    return val_loss, val_acc


def train(train_data, val_data, model, epochs=20, opt_name='AdamW': rightOptimizers,
          criterion_name='CrossEntropy': rightLosses, wandb_log=True):
    log_template = "\nEpoch {ep:03d} train_loss: {t_loss:0.4f} \
    val_loss {v_loss:0.4f} train_acc {t_acc:0.4f} val_acc {v_acc:0.4f}"

    if opt_name == 'AdamW':
        opt = torch.optim.AdamW(model.parameters(), amsgrad=True)
    elif opt_name == 'SGD':
        opt = torch.optim.SGD(model.parameters())
    elif opt_name == 'RMSprop':
        opt = torch.optim.RMSprop(model.parameters())
    elif opt_name = 'Adagrad':
        opt = torch.optim.Adagrad(model.parameters())

    if criterion_name == 'CrossEntropy':
        criterion = nn.CrossEntropyLoss()
    elif criterion_name == 'NLLLoss':
        criterion = nn.NLLLoss()
    elif criterion_name == 'KLDivLoss'
        criterion = KLDivLoss(reduction='none', log_target=False)

    name_run = str(uuid.uuid4())
        
    if wandb_log:
        config = {'balanced': True, 'epoch':epoch, 'optimizer':opt_name, 'criterion':criterion_name}
        
        run = wandb.init(project="simpsons-classifier", display_name=name_run)

    acc = 0

    with tqdm(desc="epoch", total=epochs) as pbar_outer:
        for epoch in range(epochs):
            train_loss, train_acc = fit_epoch(model, train_data, criterion, opt)
            print("loss", train_loss)
            
            val_loss, val_acc = eval_epoch(model, val_data, criterion)

            if acc > val_acc:
                torch.save(the_model.state_dict(), f'weights/{name_run}/best_model_weights_{epoch}.pth')
            
            pbar_outer.update(1)
            tqdm.write(log_template.format(ep=epoch+1, t_loss=train_loss,\
                                           v_loss=val_loss, t_acc=train_acc, v_acc=val_acc))

            if wandb_log:
                wandb.log({'val_loss': val_loss, 'val_acc': val_acc, 'train_loss': train_loss, 'train_acc': train_acc})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train model in Simpson's dataset")

    parser.add_argument('--batch', type=int, default=128)
    parser.add_argument('--optimizer', choices=['AdamW', 'SGD', 'RMSprop', 'Adagrad'], default='AdamW')
    parser.add_argument('--criterion', choices=['CrossEntropy', 'NLLLoss', 'KLDivLoss'], default='CrossEntropy')
    parser.add_argument('--epoch', type=int, default=20)
    parser.add_argument('--wandb', type=bool, default=False)
    parser.add_argument('--balanced', type=bool, default=False)
    args = parser.parse_args()

    model = MyCnn()
    train_data, val_data, test_data = get_data(args.batch, args.balanced)
    train(train_data, val_data, model, args.epoch, args.optimizer, args.criterion, args.wandb)