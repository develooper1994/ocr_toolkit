# from . import minimal_text_recognition as dtr
import argparse

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from detection.craft_text_detector.craft_text_detector.imgproc import read_image, normalize_mean_variance

try:
    # direct call
    from deep_text_recognition.demo import predict_one, logging_prediction
    from deep_text_recognition.utils import model_configuration
except:
    # indirect call
    try:
        from recognition.minimal_text_recognition.demo import predict_one, logging_prediction
        from recognition.minimal_text_recognition.utils import model_configuration, copyStateDict
    except:
        from recognition.minimal_text_recognition.demo import predict_one, logging_prediction
        from recognition.minimal_text_recognition.utils import model_configuration, copyStateDict

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class recognize:
    def __init__(self, all_opt=None):
        if all_opt is None:
            all_opt = dict(FeatureExtraction='ResNet',
                           PAD=False,
                           Prediction='Attn',
                           SequenceModeling='BiLSTM',
                           Transformation='TPS',
                           batch_max_length=25,
                           batch_size=192,
                           character='0123456789abcdefghijklmnopqrstuvwxyz',
                           hidden_size=256,
                           image_folder='demo_image/',
                           imgH=32,
                           imgW=100,
                           input_channel=1,
                           num_fiducial=20,
                           output_channel=512,
                           rgb=False,
                           saved_model='TPS-ResNet-BiLSTM-Attn.pth',
                           sensitive=False,
                           workers=4)

        # Original
        # self.models_urls = {
        #     'None-ResNet-None-CTC.pth': 'https://drive.google.com/open?id=1FocnxQzFBIjDT2F9BkNUiLdo1cC3eaO0',
        #     'None-VGG-BiLSTM-CTC.pth': 'https://drive.google.com/open?id=1GGC2IRYEMQviZhqQpbtpeTgHO_IXWetG',
        #     'None-VGG-None-CTC.pth': 'https://drive.google.com/open?id=1FS3aZevvLiGF1PFBm5SkwvVcgI6hJWL9',
        #     'TPS-ResNet-BiLSTM-Attn-case-sensitive.pth': 'https://drive.google.com/open?id=1ajONZOgiG9pEYsQ-eBmgkVbMDuHgPCaY',
        #     'TPS-ResNet-BiLSTM-Attn.pth': 'https://drive.google.com/open?id=1b59rXuGGmKne1AuHnkgDzoYgKeETNMv9',
        #     'TPS-ResNet-BiLSTM-CTC.pth': 'https://drive.google.com/open?id=1FocnxQzFBIjDT2F9BkNUiLdo1cC3eaO0',
        # }

        # my google drive
        self.models_urls = {
            'None-ResNet-None-CTC.pth': 'https://drive.google.com/file/d/1WF5XJvReLQ4DyYTbrvFzZ7zc-nFc7VGI/view?usp=sharing',
            'None-VGG-BiLSTM-CTC.pth': 'https://drive.google.com/file/d/1UcnA5eqGTj4Wq2lFp-qKOrjIbcJ0tVuP/view?usp=sharing',
            'None-VGG-None-CTC.pth': 'https://drive.google.com/file/d/1bbom7pjB37X-TqparKO4U-4cTfq01kC0/view?usp=sharing',
            'TPS-ResNet-BiLSTM-Attn-case-sensitive.pth': 'https://drive.google.com/file/d/10XlPutQuvhGR1tPgYwNAxiHNd1AvpbA9/view?usp=sharing',
            'TPS-ResNet-BiLSTM-Attn.pth': 'https://drive.google.com/file/d/1m4jUiTLFDkOhYA3ErPiPK9TGgUG_xQxz/view?usp=sharing',
            'TPS-ResNet-BiLSTM-CTC.pth': 'https://drive.google.com/file/d/1kZubKJij7hN4rERNnSd2R5e8Yfo9pinq/view?usp=sharing',
        }

        if isinstance(all_opt, dict):
            self.opt = argparse.Namespace(**all_opt)
        elif isinstance(all_opt, argparse.Namespace):
            self.opt = all_opt
        else:
            raise TypeError("Only dict and argparse.Namespace are allowed")
        self.converter, self.model = model_configuration(self.opt, download=False)

        # # prepare data. two demo images from https://github.com/bgshih/crnn#run-demo
        # AlignCollate_demo = AlignCollate(imgH=self.opt.imgH, imgW=self.opt.imgW, keep_ratio_with_pad=self.opt.PAD)
        # demo_data = RawDataset(root=self.opt.image_folder, opt=self.opt)  # use RawDataset
        # self.demo_loader = torch.utils.data.DataLoader(
        #     demo_data, batch_size=self.opt.batch_size,
        #     shuffle=False,
        #     num_workers=int(self.opt.workers),
        #     collate_fn=AlignCollate_demo, pin_memory=True)

    def __call__(self, *args, **kwargs):
        return self.make_recognization(kwargs['image_tensors'])

    # def make_recognizations(self, logging=True):
    #     self.logging = logging
    #     # predict
    #     # predict_all(self.converter, self.demo_loader, self.model, self.opt)
    #     self.model.eval()
    #     with torch.no_grad():
    #         for image_tensors, image_path_list in self.demo_loader:
    #             preds_max_prob, preds_str = self.make_recognization(image_tensors)
    #
    #             if self.logging:
    #                 logging_prediction(image_path_list, self.opt, preds_max_prob, preds_str)

    def make_recognization(self, image_tensors):
        preds, preds_str = predict_one(self.converter, image_tensors, self.model, self.opt)
        preds_prob = F.softmax(preds, dim=2)
        preds_max_prob, _ = preds_prob.max(dim=2)
        return preds_max_prob, preds_str

    def loggging(self, image_path_list, preds_max_prob, preds_str, opt):
        logging_prediction(image_path_list, preds_max_prob, preds_str, opt)

    def download_models(self):
        self.models_urls = {
            'None-ResNet-None-CTC.pth': 'https://drive.google.com/open?id=1FocnxQzFBIjDT2F9BkNUiLdo1cC3eaO0',
            'None-VGG-BiLSTM-CTC.pth': 'https://drive.google.com/open?id=1GGC2IRYEMQviZhqQpbtpeTgHO_IXWetG',
            'None-VGG-None-CTC.pth': 'https://drive.google.com/open?id=1FS3aZevvLiGF1PFBm5SkwvVcgI6hJWL9',
            'TPS-ResNet-BiLSTM-Attn-case-sensitive.pth': 'https://drive.google.com/open?id=1ajONZOgiG9pEYsQ-eBmgkVbMDuHgPCaY',
            'TPS-ResNet-BiLSTM-Attn.pth': 'https://drive.google.com/open?id=1b59rXuGGmKne1AuHnkgDzoYgKeETNMv9',
            'TPS-ResNet-BiLSTM-CTC.pth': 'https://drive.google.com/open?id=1FocnxQzFBIjDT2F9BkNUiLdo1cC3eaO0',
        }

    def __load_state_dict(self, net, weight_path):
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


if __name__ == "__main__":
    all_opt = dict(FeatureExtraction='ResNet',
                   PAD=False,
                   Prediction='Attn',
                   SequenceModeling='BiLSTM',
                   Transformation='TPS',
                   batch_max_length=25,
                   batch_size=192,
                   character='0123456789abcdefghijklmnopqrstuvwxyz',
                   hidden_size=256,
                   image_folder=r"""C:\Users\selcu\OneDrive\Desktop\idcard\idcard_crops""",
                   imgH=32,
                   imgW=100,
                   input_channel=1,
                   num_fiducial=20,
                   output_channel=512,
                   rgb=False,
                   saved_model='TPS-ResNet-BiLSTM-Attn.pth',
                   sensitive=False,
                   workers=4)

    recog = recognize(all_opt=all_opt)

    image_path = all_opt["image_folder"]
    # set image path and export folder directory
    image_name = 'crop_8.png'
    image_path = image_path + "/" + image_name
    output_dir = 'outputs/'

    # read image
    image_tensors = read_image(image_path)
    # # resize
    # img_resized, target_ratio, size_heatmap = resize_aspect_ratio(
    #     image_tensors, (all_opt["imgH"], all_opt["imgW"]), interpolation=cv2.INTER_CUBIC, mag_ratio=1  # old: cv2.INTER_LINEAR
    # )
    # ratio_h = ratio_w = 1 / target_ratio

    # preprocessing
    x = normalize_mean_variance(image_tensors)
    x = torch.from_numpy(x).permute(2, 0, 1)  # [h, w, c] to [c, h, w]
    x = x.unsqueeze(0)  # [c, h, w] to [b, c, h, w]

    image_tensors = cv2.cvtColor(image_tensors, cv2.COLOR_RGB2GRAY)
    image_tensors = np.expand_dims(image_tensors, axis=0)

    image_tensors = torch.from_numpy(image_tensors).float()  # .permute(2, 0, 1)
    image_tensors = image_tensors.unsqueeze(0)
    preds_max_prob, preds_str = recog(image_tensors=image_tensors)
    print(preds_str)
