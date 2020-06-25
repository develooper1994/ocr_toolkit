# standart modules
import os
from path import Path
import random

import numpy as np
from PIL import Image

# torch modules
import torch
from torch import optim, nn
from torch.utils.data import DataLoader
from torchvision.transforms import Resize, ToPILImage, ToTensor, Compose
from torch.optim.lr_scheduler import CosineAnnealingLR

import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter

## my modules
try:
    from .models import fully_conv_model
    from . import dataset, UFPR_ALPR_dataset
    from .evaluation import wer_eval, preds_to_integer, my_collate, AverageMeter
except:
    from models import fully_conv_model
    from dataset import UFPR_ALPR_dataset
    from evaluation import wer_eval, preds_to_integer, my_collate, AverageMeter

torch.manual_seed(0)
plt.style.use('seaborn')


# Helper to count params
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def train(epochs=5, batch_size=4, npa=1, lr=5e-4, eta_min=1e-6):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ## Set up Tensorboard writer for current test
    writer = SummaryWriter(log_dir="logs")  # /home/leander/AI/repos/OCR-CNN/logs2/correct_cosine_2
    ## Model
    model = fully_conv_model.cnn_attention_ocr(n_layers=8, nclasses=93, model_dim=64, input_dim=3)
    model = model.to(device).train()
    ctc_loss = nn.CTCLoss(blank=0, reduction="mean")
    ## Optimizer: Good initial is 5e5
    optimizer = optim.Adam(model.parameters(), lr=lr)
    cs = CosineAnnealingLR(optimizer=optimizer, T_max=250000, eta_min=eta_min)
    ## We keep track of the Average loss and CER
    ave_total_loss = AverageMeter()
    CER_total = AverageMeter()
    ## Dataset
    transforms = Compose([
        ToPILImage(),
        Resize((29, 73), Image.BICUBIC),
        ToTensor()
    ])
    root = Path(r"D:\PycharmProjects\ocr_toolkit\UFPR-ALPR dataset")
    ds = UFPR_ALPR_dataset(root, dataset_type="train", transform=transforms)
    trainset = DataLoader(ds, batch_size=batch_size)
    # testset = DataLoader(test_dataset, batch_size=batch_size)

    print(count_parameters(model))

    ## Train
    n_iter = 0
    for epochs in range(epochs):

        print("Epoch:", epochs, "started")
        for i, ge in enumerate(trainset):

            # to avoid OOM
            if ge[0].shape[3] <= 800:

                # DONT FORGET THE ZERO GRAD!!!!
                optimizer.zero_grad()

                # Get Predictions, permuted for CTC loss
                log_probs = model(ge[0].to(device)).permute((2, 0, 1))

                # Targets have to be CPU for baidu loss
                targets = ge[1].to(device)  # .cpu()

                # Get the Lengths/2 becase this is how much we downsample the width
                input_lengths = ge[2] / 2
                target_lengths = ge[3]

                # Get the CTC Loss
                input_len, batch_size, vocab_size = log_probs.size()
                log_probs_lens = torch.full(size=(batch_size,), fill_value=input_len, dtype=torch.int32)
                loss = ctc_loss(log_probs, targets, log_probs_lens, target_lengths)

                # Then backward and step
                loss.backward()
                optimizer.step()

                # Save Loss in averagemeter and write to tensorboard
                ave_total_loss.update(loss.data.item())
                writer.add_scalar("total_loss", ave_total_loss.average(), n_iter)

                # Here we Calculate the Character error rate
                # cum_len = torch.cumsum(target_lengths, axis=0)
                # targets = np.split(ge[1].cpu(), cum_len[:-1])
                targets = ge[1].cpu()
                wer_list = []
                for j in range(log_probs.shape[1]):
                    temp = log_probs[:, j, :][0:input_lengths[j], :]
                    wer = wer_eval(temp, targets[j])
                    wer_list.append(wer)

                # Here we save an example together with its decoding and truth
                # Only if it is positive

                if np.average(wer_list) > 0.1:
                    # max_value = np.max(wer_list)
                    max_elem = np.argmax(wer_list)
                    # max_image = ge[0][max_elem].cpu()
                    max_target = targets[max_elem]

                    max_target = [ds.decode_dict[x] for x in max_target.tolist()]
                    max_target = "".join(max_target)

                    ou = preds_to_integer(log_probs[:, max_elem, :])
                    max_preds = [ds.decode_dict[x] for x in ou]
                    max_preds = "".join(max_preds)

                    writer.add_text("label", max_target, n_iter)
                    writer.add_text("pred", max_preds, n_iter)
                    writer.add_image("img", ge[0][max_elem].detach().cpu().numpy(), n_iter)

                    # gen.close()
                    # break

                # Might become infinite
                if np.average(wer_list) < 10:
                    CER_total.update(np.average(wer_list))
                    writer.add_scalar("CER", CER_total.average(), n_iter)

                # We save when the new avereage CR is beloew the NPA
                # npa>CER_total.average() and CER_total.average()>0 and CER_total.average()<1
                if npa > CER_total.average() > 0 and CER_total.average() < 1:
                    torch.save(model.state_dict(), "autosave.pt")
                    npa = CER_total.average()

                n_iter = n_iter + 1
                cs.step()
                lr = optimizer.param_groups[0]["lr"]
                writer.add_scalar("lr", lr, n_iter)

        summary_string = f"||epochs: {epochs}|> " \
                         f"||n_iter: {n_iter}|> " \
                         f"||Loss: {loss.data.item()}|> " \
                         f"||max_target: {max_target}|> " \
                         f"||max_preds: {max_preds}|> " \
                         f"||Average CER total: {CER_total.average()}|> " \
                         f"||Average ave_total_loss: {ave_total_loss.average()}|> "
        print(summary_string)

    print(CER_total.average())


if __name__ == "__main__":
    epochs = 100000
    batch_size = 50
    train(epochs=epochs, batch_size=batch_size, npa=1, lr=5e-4, eta_min=1e-6)

