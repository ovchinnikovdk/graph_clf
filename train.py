import torch
from catalyst.dl import SupervisedRunner
from catalyst.dl.callbacks import F1ScoreCallback
import collections
from lib.dataset import GraphDataset
from model.gcn import GCN
from torch.utils.data import DataLoader


def train(num_epochs, model, loaders, logdir):
    criterion = torch.nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=2)

    callbacks = [F1ScoreCallback()]

    # model runner
    runner = SupervisedRunner()

    # model training
    runner.train(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        loaders=loaders,
        logdir=logdir,
        num_epochs=num_epochs,
        callbacks=callbacks,
        verbose=True
    )


def main():
    model = GCN()

    # experiment setup
    logdir = "./logdir/" + 'gcn'
    num_epochs = 10

    # data
    loaders = collections.OrderedDict()
    train_dataset = GraphDataset('data/train/')
    val_dataset = GraphDataset('data/test/')
    loaders["train"] = DataLoader(train_dataset, shuffle=True, num_workers=4, batch_size=96)
    loaders["valid"] = DataLoader(val_dataset, shuffle=False, num_workers=4, batch_size=96)

    train(num_epochs, model, loaders, logdir)


if __name__ == '__main__':
    main()
