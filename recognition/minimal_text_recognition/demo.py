import argparse
import string

import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torch.utils.data

try:
    from recognition.minimal_text_recognition.dataset import RawDataset, AlignCollate
    from recognition.minimal_text_recognition.utils import model_configuration
except:
    try:
        from dataset import RawDataset, AlignCollate
        from utils import model_configuration
    except:
        try:
            from deep_text_recognition.dataset import RawDataset, AlignCollate
            from deep_text_recognition.utils import model_configuration
        except:
            from .dataset import RawDataset, AlignCollate
            from .utils import model_configuration

# # from model import Model
# from dataset import AlignCollate, RawDataset
# from utils import model_configuration

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def demo(opts):
    """ model configuration """
    if isinstance(opts, dict):
        opt = argparse.Namespace(**opts)
    elif isinstance(opts, argparse.Namespace):
        opt = opts
    else:
        raise TypeError("Only dict and argparse.Namespace are allowed")
    converter, model = model_configuration(opt)

    # prepare data. two demo images from https://github.com/bgshih/crnn#run-demo
    AlignCollate_demo = AlignCollate(imgH=opt.imgH, imgW=opt.imgW, keep_ratio_with_pad=opt.PAD)
    demo_data = RawDataset(root=opt.image_folder, opt=opt)  # use RawDataset
    demo_loader = torch.utils.data.DataLoader(
        demo_data, batch_size=opt.batch_size,
        shuffle=False,
        num_workers=int(opt.workers),
        collate_fn=AlignCollate_demo, pin_memory=True)

    # predict
    predict_all(converter, demo_loader, model, opt)


def predict_all(converter, demo_loader, model, opt, logging=True):
    model.eval()
    with torch.no_grad():
        for image_tensors, image_path_list in demo_loader:
            preds, preds_str = predict_one(converter, image_tensors, model, opt)
            preds_prob = F.softmax(preds, dim=2)
            preds_max_prob, _ = preds_prob.max(dim=2)

            if logging:
                logging_prediction(image_path_list, preds_max_prob, preds_str, opt)


def predict_one(converter, image_tensors, model, opt):
    try:
        batch_size = image_tensors.size(0)
    except:
        batch_size = 1
    image = image_tensors.to(device)
    # For max length prediction
    length_for_pred = torch.IntTensor([opt.batch_max_length] * batch_size).to(device)
    text_for_pred = torch.LongTensor(batch_size, opt.batch_max_length + 1).fill_(0).to(device)
    if 'CTC' in opt.Prediction:
        preds = model(image, text_for_pred)

        # Select max probabilty (greedy decoding) then decode index to character
        preds_size = torch.IntTensor([preds.size(1)] * batch_size)
        _, preds_index = preds.max(2)
        preds_index = preds_index.view(-1)
        preds_str = converter.decode(preds_index.data, preds_size.data)

    else:
        preds = model(image, text_for_pred, is_train=False)

        # select max probabilty (greedy decoding) then decode index to character
        _, preds_index = preds.max(2)
        preds_str = converter.decode(preds_index, length_for_pred)
    return preds, preds_str


def logging_prediction(image_path_list, preds_max_prob, preds_str, opt):
    log = open(f'./log_demo_result.txt', 'a')
    dashed_line = '-' * 80
    head = f'{"image_path":25s}\t{"predicted_labels":25s}\tconfidence score'
    print(f'{dashed_line}\n{head}\n{dashed_line}')
    log.write(f'{dashed_line}\n{head}\n{dashed_line}\n')
    for img_name, pred, pred_max_prob in zip(image_path_list, preds_str, preds_max_prob):
        if 'Attn' in opt.Prediction:
            pred_EOS = pred.find('[s]')
            pred = pred[:pred_EOS]  # prune after "end of sentence" token ([s])
            pred_max_prob = pred_max_prob[:pred_EOS]

        # calculate confidence score (= multiply of pred_max_prob)
        confidence_score = pred_max_prob.cumprod(dim=0)[-1]

        print(f'{img_name:25s}\t{pred:25s}\t{confidence_score:0.4f}')
        log.write(f'{img_name:25s}\t{pred:25s}\t{confidence_score:0.4f}\n')
    log.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_folder', type=str, required=True, default='demo_image/',
                        help='path to image_folder which contains text images')
    parser.add_argument('--workers', type=int, default=4,
                        help='number of data loading workers')
    parser.add_argument('--batch_size', type=int, default=192,
                        help='input batch size')
    parser.add_argument('--saved_model', type=str, required=True, default='TPS-ResNet-BiLSTM-Attn.pth',
                        help="path to saved_model to evaluation")
    """ Data processing """
    parser.add_argument('--batch_max_length', type=int, default=25,
                        help='maximum-label-length')
    parser.add_argument('--imgH', type=int, default=32,
                        help='the height of the input image')
    parser.add_argument('--imgW', type=int, default=100,
                        help='the width of the input image')
    parser.add_argument('--rgb', action='store_true', default=False,
                        help='use rgb input')
    parser.add_argument('--character', type=str, default='0123456789abcdefghijklmnopqrstuvwxyz',
                        help='character label')
    parser.add_argument('--sensitive', action='store_true', default=False,
                        help='for sensitive character mode')
    parser.add_argument('--PAD', action='store_true', default=False,
                        help='whether to keep ratio then pad for image resize')
    """ Model Architecture """
    parser.add_argument('--Transformation', type=str, required=True, default='TPS',
                        help='Transformation stage. None|TPS')
    parser.add_argument('--FeatureExtraction', type=str, required=True, default='ResNet',
                        help='FeatureExtraction stage. VGG|RCNN|ResNet')
    parser.add_argument('--SequenceModeling', type=str, required=True, default='BiLSTM',
                        help='SequenceModeling stage. None|BiLSTM')
    parser.add_argument('--Prediction', type=str, required=True, default="Attn",
                        help='Prediction stage. CTC|Attn')
    parser.add_argument('--num_fiducial', type=int, default=20,
                        help='number of fiducial points of TPS-STN')
    parser.add_argument('--input_channel', type=int, default=1,
                        help='the number of input channel of Feature extractor')
    parser.add_argument('--output_channel', type=int, default=512,
                        help='the number of output channel of Feature extractor')
    parser.add_argument('--hidden_size', type=int, default=256,
                        help='the size of the LSTM hidden state')

    opt = parser.parse_args()
    print(opt)
    print(type(opt))

    """ vocab / character number configuration """
    if opt.sensitive:
        opt.character = string.printable[:-6]  # same with ASTER setting (use 94 char).

    cudnn.benchmark = True
    cudnn.deterministic = True
    opt.num_gpu = torch.cuda.device_count()

    all_opt = dict(FeatureExtraction='ResNet',
                   PAD=False,
                   Prediction='Attn',
                   SequenceModeling='BiLSTM',
                   Transformation='TPS',
                   batch_max_length=25,
                   batch_size=192,
                   character='0123456789abcdefghijklmnopqrstuvwxyz',
                   hidden_size=256,
                   image_folder='/home/selcuk/Desktop/plates/plate6_crops/',
                   imgH=32,
                   imgW=100,
                   input_channel=1,
                   num_fiducial=20,
                   output_channel=512,
                   rgb=False,
                   saved_model='TPS-ResNet-BiLSTM-Attn.pth',
                   sensitive=False,
                   workers=4)

    # demo(opt)
    demo(all_opt)
