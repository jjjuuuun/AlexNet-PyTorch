from data import TrainDataset
import torch
from torch.utils.data import DataLoader, random_split
import torch.nn.functional as F
from pathlib import Path
from models import AlexNet, accuracy
from torch import optim
from torchsummary import summary
from torch import nn
import argparse
import os
import wandb
import time


def main(opt):

    if opt.use_wandb:
        print("Using wandb ...")
        wandb.init(project=opt.wandb_project, entity=opt.wandb_entity)
        wandb.config.update(opt)

    data_path = Path(os.path.abspath(opt.data_path))
    save_path = Path(os.path.abspath(opt.save_path))

    if not save_path.exists():
        print('Make save_path')
        save_path.mkdir()

    print(f'Directory of datasets >> {data_path}')
    print(f'Directory of saved weight >> {save_path}')

    dataset = TrainDataset(data_path)
    train_set, val_set = random_split(dataset, [opt.train_val_split, 1-opt.train_val_split])

    print(f'The number of training images = {len(train_set)}')
    print(f'The number of validation images = {len(val_set)}')

    train_iter = DataLoader(train_set,
                        batch_size=opt.batch_size,
                        shuffle = True)
    val_iter = DataLoader(val_set,
                        batch_size=opt.batch_size,
                        shuffle = True)

        
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f'The device is ready\t>>\t{device}')

    model = AlexNet(num_classes = opt.num_classes).to(device)
    print('The model is ready ...')
    print(summary(model, (3, 224, 224)))

    ########## Original paper code #########
    # optimizer = optim.SGD(
    #     params = model.parameters(),
    #     lr = opt.learning_rate,
    #     momentum = opt.momentum,
    #     weight_decay = opt.weight_decay
    # )

    # lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 
    #                                                     mode=opt.ls_mode, 
    #                                                     factor=opt.ls_factor,
    #                                                     threshold=opt.ls_threshold,
    #                                                     verbose=True)
    ########################################

    ########## Custom optimizer & learning schduler code #########
    optimizer = optim.Adam(params=model.parameters(), lr=opt.learning_rate)
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
    ##############################################################

    criterion = nn.CrossEntropyLoss()

    print("Starting training ...")

    if opt.use_wandb:
        wandb.watch(model)

    for epoch in range(opt.epochs):
        start_time = time.time()

        train_epoch_loss = 0
        model.train()
        for train_img, train_target in train_iter:
            train_img, train_target = train_img.to(device), train_target.to(device)
            
            optimizer.zero_grad()

            train_pred = model(train_img)
            train_iter_loss = criterion(train_pred, train_target)
            train_iter_loss.backward()
            optimizer.step()

            train_epoch_loss += train_iter_loss

        train_epoch_loss = train_epoch_loss / len(train_iter)
        lr_scheduler.step()

        # Validation
        with torch.no_grad():
            val_epoch_loss = 0
            model.eval()
            for val_img, val_target in val_iter:
                val_img, val_target = val_img.to(device), val_target.to(device)

                val_pred = model(val_img)
                val_iter_loss = criterion(val_pred, val_target).detach()

                val_epoch_loss += val_iter_loss
            model.train()

        ####### If you are using ReduceLROnPlateau #######
        # lr_scheduler.step(val_epoch_loss)
        ##################################################
        val_epoch_loss = val_epoch_loss / len(val_iter)

        train_acc = accuracy(model, train_iter, device)
        val_acc = accuracy(model, val_iter, device)

        print('time >> {:.4f}\tepoch >> {:04d}\ttrain_acc >> {:.4f}\ttrain_loss >> {:.4f}\tval_acc >> {:.4f}\tval_loss >> {:.4f}'
              .format(time.time()-start_time, epoch, train_acc, train_epoch_loss, val_acc, val_epoch_loss))
        
        if (epoch+1) % 5 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                }, save_path / f'epoch({epoch})_acc({train_acc:.3f})_loss({train_epoch_loss:.3f}).pt')
        
        if opt.use_wandb:
            wandb.log({'train_acc': train_acc,
                    'train_loss': train_epoch_loss,
                    'val_acc': val_acc,
                    'val_loss': val_epoch_loss})


if __name__ == '__main__':
    args = argparse.ArgumentParser(description='AlexNet Implementation')
    args.add_argument('--data_path', default=None, type=str, required=True,
                        help='path of datasets(.zip or dir), type: str -> pathlib')
    args.add_argument('--save_path', default=None, type=str, required=True,
                        help='path to save weight')
    args.add_argument('--batch_size', default=128, type=int,
                        help='batch size of training')
    args.add_argument('--epochs', default=100, type=int,
                        help='epoch of training')
    args.add_argument('--num_classes', default=2, type=int,
                        help='number of classes')
    args.add_argument('--train_val_split', default=0.8, type=float,
                    help='ratio of train_set and val_set')
    args.add_argument('--is_train', default=True, type=bool,
                        help='Train or Test (True or False)')
    args.add_argument('--learning_rate', default=0.0001, type=float,
                        help='optimizer learning_rate')
    args.add_argument('--momentum', default=0.9, type=float,
                        help='optimizer momentum')
    args.add_argument('--weight_decay', default=0.0005, type=float,
                        help='optimizer momentum')
    args.add_argument('--ls_factor', default=0.1, type=float,
                        help='learning scheduler factor')
    args.add_argument('--ls_mode', default='min', type=str,
                        help='learning scheduler mode')
    args.add_argument('--ls_threshold', default=1e-4, type=float,
                        help='learning scheduler threshold')
    args.add_argument('--use_wandb', default=False, type=bool,
                        help='Using wandb')
    args.add_argument('--wandb_project', default=None, type=str,
                        help='wandb project')
    args.add_argument('--wandb_entity', default=None, type=str,
                        help='wandb entity')
    

    opt = args.parse_args()

    if opt.is_train:
        main(opt)