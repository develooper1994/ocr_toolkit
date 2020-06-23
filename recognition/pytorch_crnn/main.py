# standart modules
import os
import torch

# torch modules
from torch import optim, nn
from torch.utils.data import DataLoader
from catalyst import dl
from catalyst.contrib.data.transforms import ToTensor
from catalyst.contrib.datasets import MNIST
from catalyst.utils import metrics

## my modules
try:
    from recognition.pytorch_crnn.models import fully_conv_model
except:
    import fully_conv_model


class crnn_trainer(dl.Runner):

    def _handle_batch(self, batch):
        x, targets = batch
        x_noise = (x + torch.rand_like(x)).clamp_(0, 1)
        log_probs = self.model(x_noise)

        input_len, batch_size, vocab_size = log_probs.size()
        target_lengths = len(targets)
        log_probs_lens = torch.full(size=(batch_size,), fill_value=input_len, dtype=torch.int32)
        loss = ctc_loss(log_probs, targets, log_probs_lens.cuda(), target_lengths)
        accuracy01, accuracy03, accuracy05 = metrics.accuracy(log_probs, targets, topk=(1, 3, 5))
        self.batch_metrics = {
            "loss": loss,
            "accuracy01": accuracy01,
            "accuracy03": accuracy03,
            "accuracy05": accuracy05,
        }

        if self.is_train_loader:
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()


if __name__ == "__main__":
    model = fully_conv_model.cnn_attention_ocr(n_layers=8, nclasses=93, model_dim=64, input_dim=1)
    # cnn = cnn.cuda().train()
    optimizer = optim.AdamW(model.parameters(), lr=0.02)
    ctc_loss = nn.CTCLoss(blank=0)

    loaders = {
        "train": DataLoader(MNIST(os.getcwd(), train=False, download=True, transform=ToTensor()), batch_size=32),
        "valid": DataLoader(MNIST(os.getcwd(), train=False, download=True, transform=ToTensor()), batch_size=32),
    }

    runner = crnn_trainer()
    runner.train(
        model=model,
        optimizer=optimizer,
        loaders=loaders,
        verbose=True,
    )
