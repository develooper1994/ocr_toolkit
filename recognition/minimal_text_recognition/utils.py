import os
import warnings
from collections import OrderedDict
from pathlib import Path

import torch

# from detection.craft_text_detector.craft_text_detector import file_utils

try:
    from model import Model
except:
    try:
        from .model import Model
    except:
        from recognition.minimal_text_recognition.model import Model

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class CTCLabelConverter(object):
    """ Convert between text-label and text-index """

    def __init__(self, character):
        # character (str): set of the possible characters.
        dict_character = list(character)

        self.dict = {}
        for i, char in enumerate(dict_character):
            # NOTE: 0 is reserved for 'blank' token required by CTCLoss
            self.dict[char] = i + 1

        self.character = ['[blank]'] + dict_character  # dummy '[blank]' token for CTCLoss (index 0)

    def encode(self, text, batch_max_length=25):
        """convert text-label into text-index.
        input:
            text: text labels of each image. [batch_size]

        output:
            text: concatenated text index for CTCLoss.
                    [sum(text_lengths)] = [text_index_0 + text_index_1 + ... + text_index_(n - 1)]
            length: length of each text. [batch_size]
        """
        length = [len(s) for s in text]
        text = ''.join(text)
        text = [self.dict[char] for char in text]

        return (torch.IntTensor(text), torch.IntTensor(length))

    def decode(self, text_index, length):
        """ convert text-index into text-label. """
        texts = []
        index = 0
        for l in length:
            t = text_index[index:index + l]

            char_list = []
            for i in range(l):
                if t[i] != 0 and (not (i > 0 and t[i - 1] == t[i])):  # removing repeated characters and blank.
                    char_list.append(self.character[t[i]])
            text = ''.join(char_list)

            texts.append(text)
            index += l
        return texts


class AttnLabelConverter(object):
    """ Convert between text-label and text-index """

    def __init__(self, character):
        # character (str): set of the possible characters.
        # [GO] for the start token of the attention decoder. [s] for end-of-sentence token.
        list_token = ['[GO]', '[s]']  # ['[s]','[UNK]','[PAD]','[GO]']
        list_character = list(character)
        self.character = list_token + list_character

        self.dict = {}
        for i, char in enumerate(self.character):
            # print(i, char)
            self.dict[char] = i

    def encode(self, text, batch_max_length=25):
        """ convert text-label into text-index.
        input:
            text: text labels of each image. [batch_size]
            batch_max_length: max length of text label in the batch. 25 by default

        output:
            text : the input of attention decoder. [batch_size x (max_length+2)] +1 for [GO] token and +1 for [s] token.
                text[:, 0] is [GO] token and text is padded with [GO] token after [s] token.
            length : the length of output of attention decoder, which count [s] token also. [3, 7, ....] [batch_size]
        """
        length = [len(s) + 1 for s in text]  # +1 for [s] at end of sentence.
        # batch_max_length = max(length) # this is not allowed for multi-gpu setting
        batch_max_length += 1
        # additional +1 for [GO] at first step. batch_text is padded with [GO] token after [s] token.
        batch_text = torch.LongTensor(len(text), batch_max_length + 1).fill_(0)
        for i, t in enumerate(text):
            text = list(t)
            text.append('[s]')
            text = [self.dict[char] for char in text]
            batch_text[i][1:1 + len(text)] = torch.LongTensor(text)  # batch_text[:, 0] = [GO] token
        return (batch_text.to(device), torch.IntTensor(length).to(device))

    def decode(self, text_index, length):
        """ convert text-index into text-label. """
        texts = []
        for index, l in enumerate(length):
            text = ''.join([self.character[i] for i in text_index[index, :]])
            texts.append(text)
        return texts


class Averager(object):
    """Compute average for torch.Tensor, used for loss average."""

    def __init__(self):
        self.reset()

    def add(self, v):
        count = v.data.numel()
        v = v.data.sum()
        self.n_count += count
        self.sum += v

    def reset(self):
        self.n_count = 0
        self.sum = 0

    def val(self):
        res = 0
        if self.n_count != 0:
            res = self.sum / float(self.n_count)
        return res


def model_configuration(opt, model_path=None, model_url=None, download=False):
    if 'CTC' in opt.Prediction:
        converter = CTCLabelConverter(opt.character)
    else:
        converter = AttnLabelConverter(opt.character)
    opt.num_class = len(converter.character)
    if opt.rgb:
        opt.input_channel = 3
    model = Model(opt)
    print('model input parameters', opt.imgH, opt.imgW, opt.num_fiducial, opt.input_channel, opt.output_channel,
          opt.hidden_size, opt.num_class, opt.batch_max_length, opt.Transformation, opt.FeatureExtraction,
          opt.SequenceModeling, opt.Prediction)
    # model = torch.nn.DataParallel(model).to(device)
    # load model

    if model_path is None:
        warnings.warn("models should be same directory as the image directory")
        model_path = opt.image_folder
    net_name = opt.saved_model
    print('loading pretrained model from %s' % net_name)
    if model_url is None:
        models_urls = {
            'None-ResNet-None-CTC.pth': 'https://drive.google.com/file/d/1WF5XJvReLQ4DyYTbrvFzZ7zc-nFc7VGI/view?usp=sharing',
            'None-VGG-BiLSTM-CTC.pth': 'https://drive.google.com/file/d/1UcnA5eqGTj4Wq2lFp-qKOrjIbcJ0tVuP/view?usp=sharing',
            'None-VGG-None-CTC.pth': 'https://drive.google.com/file/d/1bbom7pjB37X-TqparKO4U-4cTfq01kC0/view?usp=sharing',
            'TPS-ResNet-BiLSTM-Attn-case-sensitive.pth': 'https://drive.google.com/file/d/10XlPutQuvhGR1tPgYwNAxiHNd1AvpbA9/view?usp=sharing',
            'TPS-ResNet-BiLSTM-Attn.pth': 'https://drive.google.com/file/d/1m4jUiTLFDkOhYA3ErPiPK9TGgUG_xQxz/view?usp=sharing',
            'TPS-ResNet-BiLSTM-CTC.pth': 'https://drive.google.com/file/d/1kZubKJij7hN4rERNnSd2R5e8Yfo9pinq/view?usp=sharing',
        }
        model_url = models_urls[net_name]
    if download:
        raise NotImplementedError  #  not working correctly.
        # TODO! implement with pycurl to download big models. like... demo.ipynb
        # models = {
        # 'None-ResNet-None-CTC.pth': 'https://drive.google.com/open?id=1FocnxQzFBIjDT2F9BkNUiLdo1cC3eaO0',
        # 'None-VGG-BiLSTM-CTC.pth': 'https://drive.google.com/open?id=1GGC2IRYEMQviZhqQpbtpeTgHO_IXWetG',
        # 'None-VGG-None-CTC.pth': 'https://drive.google.com/open?id=1FS3aZevvLiGF1PFBm5SkwvVcgI6hJWL9',
        # 'TPS-ResNet-BiLSTM-Attn-case-sensitive.pth': 'https://drive.google.com/open?id=1ajONZOgiG9pEYsQ-eBmgkVbMDuHgPCaY',
        # 'TPS-ResNet-BiLSTM-Attn.pth': 'https://drive.google.com/open?id=1b59rXuGGmKne1AuHnkgDzoYgKeETNMv9',
        # 'TPS-ResNet-BiLSTM-CTC.pth': 'https://drive.google.com/open?id=1FocnxQzFBIjDT2F9BkNUiLdo1cC3eaO0',
        # }
        #
        # for k, v in models.items():
        # doc_id = v[v.find('=')+1:]
        # !curl -c /tmp/cookies "https://drive.google.com/uc?export=download&id=$doc_id" > /tmp/intermezzo.html
        # !curl -L -b /tmp/cookies "https://drive.google.com$(cat /tmp/intermezzo.html | grep -Po 'uc-download-link" [^>]* href="\K[^"]*' | sed 's/\&amp;/\&/g')" > $k
        #
        # !ls -al *.pth
        model_path = get_weights(model_path=model_path, model_url=model_url, net_name=net_name)
    # model.load_state_dict(copyStateDict(net_name, map_location=device))  # error!
    try:
        model = __load_state_dict(model, model_path + "/" + net_name)
    except:
        model = __load_state_dict(model, net_name)
    return converter, model


def get_weights(model_path=None, model_url=None, net_name: str = "/TPS-ResNet-BiLSTM-Attn.pth"):
    """
    Downloads weights and biases if model_path is empty.
        Default download path:
            Linux: $HOME/.craft_text_detector/weights
            Windows: $HOME/.craft_text_detector/weights
    :param model_path: Serialized network(model) file
    :param model_url: network(model) url
    :param net_name: network(model) file name
    :type net_name: str
    :return: weight path
        if model_path is None:
            weight_path = "$HOME/.craft_text_detector/weights"
    """
    home_path = str(Path.home())
    if model_path is None:
        weight_path = os.path.join(
            home_path, ".handwritten_recognition", "weights", net_name
        )
    else:
        weight_path = model_path

    # check if weights are already downloaded. if not, download
    if os.path.isfile(weight_path) is not True:
        # download to given weight_path
        print("Craft text detector weight will be downloaded to {}".format(weight_path))
        # file_utils.download(url=model_url, save_path=weight_path)  # TODO! gdown can't download large pretranied models. Use curl
        # file_utils.download_model_from_google_drive(url=model_url, save_path=model_path+"/"+net_name)  # TODO! gdown can't download large pretranied models. Use curl
    else:
        weight_path = model_path
    return weight_path


def copyStateDict(state_dict):
    """
    Copies network(model) deserialized weights and biases.
    :param state_dict: Deserialized weights and biases
    :return: New deserialized weights and biases
    """
    if list(state_dict.keys())[0].startswith("module"):
        start_idx = 1
    else:
        start_idx = 0
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = ".".join(k.split(".")[start_idx:])
        new_state_dict[name] = v
    return new_state_dict


def __load_state_dict(net, weight_path, device="cpu"):
    """
    1) Loads weights and biases.
    2) Deserialize them.
    3) Transport to cuda
    4) Make it pytorch "dataparallel"
    5) Turn it into evaluation mode.
    6) Return it.
    :param net: Artificial Neural network(model) that makes main job
    :param weight_path: Serialized pth file path with name
    :return: loaded network
    """
    net.load_state_dict(copyStateDict(torch.load(weight_path)))

    net = net.to(device)
    net = torch.nn.DataParallel(net)
    net.eval()
    return net
