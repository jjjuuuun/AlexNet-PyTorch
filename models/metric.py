import torch


def accuracy(model,data_iter,device):
    with torch.no_grad():
        n_total, n_correct = 0, 0
        model.eval() # evaluate (affects DropOut and BN)
        for data, target in data_iter:
            y_target = target.data.view(-1).to(device)
            model_pred = model(data.to(device))
            _, y_pred = torch.max(model_pred.data, 1)
            n_correct += (y_pred==y_target).sum().item()
            n_total += data.size(0)
        acc = (n_correct/n_total)
        model.train() # back to train mode 

    return acc