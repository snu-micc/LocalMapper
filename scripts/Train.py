from argparse import ArgumentParser

import torch
import sklearn
import torch.nn as nn

from utils import mkdir_p, get_user_name, init_featurizer, get_configure, load_model, load_dataloader, predict

def run_a_train_epoch(args, epoch, model, data_loader, loss_criterion, optimizer):
    model.train()
    train_loss = 0
    for batch_id, batch_data in enumerate(data_loader):
        idxs, rxns, rbg, pbg, radms, padms, labels_list, masks_list = batch_data
        labels_list, masks_list = [labels.to(args['device']) for labels in labels_list], [masks.to(args['device']) for masks in masks_list]
        logits_list = predict(args, model, rbg, pbg, radms, padms)
        loss = 0
        for logits, labels, masks in zip(logits_list, labels_list, masks_list):
            loss += (loss_criterion(logits, labels) * (masks != 0)).float().mean()
        loss = loss/args['batch_size']
        optimizer.zero_grad()      
        loss.backward() 
        nn.utils.clip_grad_norm_(model.parameters(), args['max_clip'])
        optimizer.step()
        train_loss += loss.item()    
        if batch_id % args['print_every'] == 0:
            print('\repoch %d/%d, batch %d/%d, loss %.4f' % (epoch + 1,args['num_epochs'],batch_id+1,len(data_loader),loss), end='', flush=True)
#             print('\repoch %d/%d, batch %d/%d, loss %.4f, lr: %.6f' % (epoch + 1, args['num_epochs'], batch_id+1, len(data_loader), loss, scheduler.get_last_lr()[0]), end='', flush=True)
#     scheduler.step()
    return


def run_an_val_epoch(args, model, data_loader, loss_criterion):
    model.eval()
    temp_list,val_loss = [],0
    rxns_list, mapped_rxns_list = [], []
        
    with torch.no_grad():
        for batch_id, batch_data in enumerate(data_loader):
            idxs, rxns, rbg, pbg, radms, padms, labels_list, masks_list = batch_data
            labels_list, masks_list = [labels.to(args['device']) for labels in labels_list], [masks.to(args['device']) for masks in masks_list]
            logits_list = predict(args, model, rbg, pbg, radms, padms)
            loss = 0
            for logits, labels, masks in zip(logits_list, labels_list, masks_list):
                loss += (loss_criterion(logits, labels) * (masks != 0)).float().mean()
            loss = loss/args['batch_size']
            val_loss += loss.item()  
    return val_loss/(batch_id+1)

def main(args):
    args['mode'] = 'train'
    args['chemist_name'] = get_user_name(args)
    args['device'] = torch.device(args['gpu']) if torch.cuda.is_available() else torch.device('cpu')
    print ('Trianing with device %s, chemist name: %s' % (args['device'], args['chemist_name']))
        
    model_name = 'LocalMapper_%d.pth' % (args['iteration'])
    args['data_dir'] = '../data/%s/' % args['dataset']
    args['model_dir'] = '../models/%s/%s/' % (args['dataset'], args['chemist_name'])
    args['model_path'] = args['model_dir'] + model_name
    mkdir_p(args['model_dir'])
    
    args = init_featurizer(args)
    args['config_path'] = '../data/configs/%s' % args['config']
    train_loader, val_loader = load_dataloader(args)
    model, loss_criterion, optimizer, scheduler, stopper = load_model(args)
    for epoch in range(args['num_epochs']):
        run_a_train_epoch(args, epoch, model, train_loader, loss_criterion, optimizer)
        if args['iteration'] == 1:
            val_loss = 1/(epoch+1)
        else:
            val_loss = run_an_val_epoch(args, model, val_loader, loss_criterion)
            print(', validation loss: %.4f' % val_loss)
        early_stop = stopper.step(val_loss, model)
        scheduler.step(val_loss)
        if early_stop:
            print ('Model is Early stopped!!')
            break

    
if __name__ == '__main__':
    parser = ArgumentParser('LocalMapper training arguements')
    parser.add_argument('-g', '--gpu', default='cuda:0', help='GPU device to use')
    parser.add_argument('-d', '--dataset', default='USPTO_50K', help='Dataset to use')
    parser.add_argument('-c', '--config', default='default_config.json', help='Configuration of model')
    parser.add_argument('-b', '--batch-size', default=20, help='Batch size of dataloader')                             
    parser.add_argument('-n', '--num-epochs', type=int, default=30, help='Maximum number of epochs for training')
    parser.add_argument('-p', '--patience', type=int, default=5, help='Patience for early stopping')
    parser.add_argument('-i', '--iteration', type=int, default=1, help='Iteration of active learning')
    parser.add_argument('-cl', '--max-clip', type=int, default=20, help='Maximum number of gradient clip')
    parser.add_argument('-lr', '--learning-rate', type=float, default=1e-3, help='Learning rate of optimizer')
    parser.add_argument('-l2', '--weight-decay', type=float, default=1e-6, help='Weight decay of optimizer')
    parser.add_argument('-ss', '--schedule_step', type=int, default=1, help='Step size of learning scheduler')
    parser.add_argument('-pe', '--print-every', type=int, default=20, help='Print the training progress every X mini-batches')
    args = parser.parse_args().__dict__
    
#     print('Using device %s' % args['device'])
    main(args)