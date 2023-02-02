import os
from tqdm import tqdm
from PIDNet.utils.function import train, validate
import torch.backends.cudnn as cudnn
from torch.optim import SGD
from PIDNet.utils.utils import FullModel
from PIDNet.utils.criterion import BondaryLoss, OhemCrossEntropy, CrossEntropy
from PIDNet.models.pidnet import get_pred_model, get_seg_model
import torch
from matplotlib import pyplot as plt
import numpy as np
from dataset import CarlaDataset

if __name__ == '__main__':
    train_dataset = CarlaDataset(["data/Town05",
                                  "data/Town07",
                                  "data/Town03"])

    test_dataset = CarlaDataset(["data/Town10"])

    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=8, shuffle=True)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=8, shuffle=True)
    model = get_seg_model(model_name='test',
                          path_to_pretrained='checkpoints/best.pt',
                          num_classes=23, imgnet_pretrained=True)

    device = torch.device('cuda')
    class_weights = torch.FloatTensor([0.8373, 0.918, 0.866, 1.0345,
                                       1.0166, 0.9969, 0.9754, 1.0489,
                                       0.8786, 1.0023, 0.9539, 0.9843,
                                       1.1116, 0.9037, 1.0865, 1.0955,
                                       1.0865, 1.1529, 1.0507, 1,
                                       1, 1, 1]).to(device)

    sem_criterion = OhemCrossEntropy(ignore_label=-1,
                                     thres=0.9,
                                     min_kept=131072,
                                     weight=class_weights)
    # sem_criterion = CrossEntropy(ignore_label=-1, weight=class_weights)

    bd_criterion = BondaryLoss()
    model = FullModel(model, sem_criterion, bd_criterion).to(device)
    optimizer = SGD(model.parameters(), lr=0.005)

    EPOCHS = 200
    epoch_iters = len(train_dataloader)
    num_iters = EPOCHS * epoch_iters
    cudnn.benchmark = True
    cudnn.deterministic = False
    cudnn.enabled = True

    best_mIoU = 0
    real_end = 200

    for epoch in range(EPOCHS):
        train(epoch=epoch + 1, num_epoch=EPOCHS, epoch_iters=len(train_dataloader), base_lr=0.01, num_iters=num_iters,
              trainloader=train_dataloader, optimizer=optimizer, model=model)
        valid_loss, mean_IoU, IoU_array = validate(test_dataloader, model)

        if mean_IoU > best_mIoU:
            best_mIoU = mean_IoU
            torch.save(model.model.state_dict(),
                       os.path.join('checkpoints', 'best.pt'))
