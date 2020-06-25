# standart modules
import os
from path import Path

import numpy as np
from PIL import Image

# torch modules
import torch
from torch import optim, nn
from torch.utils.data import DataLoader
from torchvision.transforms import Resize, ToPILImage, ToTensor
from catalyst import dl
from catalyst.contrib.data.transforms import Compose  # ToTensor
from catalyst.contrib.datasets import MNIST
from catalyst.utils import metrics

## my modules
try:
    from .models import fully_conv_model
    from . import dataset, UFPR_ALPR_dataset
    from .evaluation import wer_eval, preds_to_integer, my_collate, AverageMeter
except:
    from models import fully_conv_model
    from dataset import UFPR_ALPR_dataset
    from evaluation import wer_eval, preds_to_integer, my_collate, AverageMeter


class crnn_trainer(dl.Runner):
    def _handle_batch(self, batch):
        x, targets, input_lengths, target_lengths = batch
        x_noise = (x + torch.rand_like(x)).clamp_(0, 1)
        log_probs = self.model(x_noise).permute((2, 0, 1))

        input_len, batch_size, vocab_size = log_probs.size()
        # target_lengths = torch.tensor(len(targets[0]))
        log_probs_lens = torch.full(size=(batch_size,), fill_value=input_len, dtype=torch.int32)
        loss = ctc_loss(log_probs, targets, log_probs_lens.cuda(), target_lengths)  # RuntimeError: target_lengths must be of size batch_size. Attetion gives shape differently

        # # Here we Calculate the Character error rate
        # cum_len = torch.cumsum(target_lengths, axis=0)
        # targets = targets.cpu()
        # wer_list = []
        # input_lengths = input_lengths/2
        # for j in range(log_probs.shape[1]):
        #     wer_list.append(wer_eval(log_probs[:, j, :][0:input_lengths[j], :], targets[j]))

        # accuracy01, accuracy03, accuracy05 = metrics.accuracy(log_probs.permute((1,2,0)), targets, topk=(1, 3, 5))  # RuntimeError: The size of tensor a (8) must match the size of tensor b (94) at non-singleton dimension 2
        self.batch_metrics = {
            "loss": loss,
        }

        if self.is_train_loader:
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()


if __name__ == "__main__":
    model = fully_conv_model.cnn_attention_ocr(n_layers=8, nclasses=93, model_dim=64, input_dim=3)
    # cnn = cnn.cuda().train()
    optimizer = optim.AdamW(model.parameters(), lr=0.02)
    ctc_loss = nn.CTCLoss(blank=0)

    num_epochs = 5
    batch_size = 10
    # train_dataset = MNIST(root, train=False, download=True, transform=ToTensor())
    # test_dataset = MNIST(root, train=False, download=True, transform=ToTensor())
    transforms = Compose([
        ToPILImage(),
        Resize((29, 73), Image.BICUBIC),
        ToTensor()
    ])
    root = Path(r"D:\PycharmProjects\ocr_toolkit\UFPR-ALPR dataset")
    train_dataset = UFPR_ALPR_dataset(root, dataset_type="train", transform=transforms)
    test_dataset = UFPR_ALPR_dataset(root, dataset_type="valid", transform=transforms)
    loaders = {
        "train": DataLoader(train_dataset, batch_size=batch_size),
        "valid": DataLoader(test_dataset, batch_size=batch_size),
    }

    runner = crnn_trainer()
    runner.train(
        model=model,
        num_epochs=num_epochs,
        optimizer=optimizer,
        loaders=loaders,
        verbose=True,
    )
