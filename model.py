# This code is written by Jingyuan Yang @ XD

from __future__ import absolute_import
from __future__ import division

import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from models.resnet import ResNet18
# from models.ResNet_18 import resnet18
from models.Decoder import Decoder
# from models.Decoder_aa import Decoder_aa
# from models.Decoder_reviewer import Decoder_reviewer

class model_resnet(nn.Module):
    """ResNet50 for Visual Sentiment Analysis on FI_8"""
    # """ResNet50 for Visual Sentiment Analysis on flickr_2"""

    def __init__(self, base_model):
        super(model_resnet, self).__init__()
        self.fcn = nn.Sequential(*list(base_model.children())[:-2])
        # self.sal = resnet18(pretrained=True)
        self.face = ResNet18()
        # self.pooling = nn.MaxPool2d((1, 2))
        self.conv1 = nn.Conv1d(3072, 3072, 1, bias=True)
        self.lstm = Decoder(feat_size=2048, hidden_size=512)
        self.sigmoid = nn.Sigmoid()

        self.GAvgPool = nn.AvgPool2d(kernel_size=14)

        self.classifiers_x = nn.Sequential(
            # nn.Dropout(p=0.5),
            nn.Linear(in_features=2048, out_features=128),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(in_features=128, out_features=1)
            )

        self.classifiers_rcnn = nn.Sequential(
            # nn.Dropout(p=0.5),
            nn.Linear(in_features=512, out_features=128),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(in_features=128, out_features=1)
            )

        self.classifiers_face = nn.Sequential(
            # nn.Dropout(p=0.5),
            nn.Linear(in_features=512, out_features=128),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(in_features=128, out_features=1)
            )

        self.classifiers8_2 = nn.Sequential(
            # nn.Dropout(p=0.5),
            nn.Linear(in_features=2048 + 512 + 512, out_features=128),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(in_features=128, out_features=8)
            # nn.Dropout(p=0.75)
            )

        self.classifier8 = nn.Sequential(
            # nn.Dropout(p=0.5),
            nn.Linear(in_features=2048 + 512  +512, out_features=8)
            )

        self.classifier2 = nn.Sequential(
            # nn.Dropout(p=0.5),
            nn.Linear(in_features=2048 + 512 + 512, out_features=2)
        )


        self.ReLU = nn.ReLU()
        self.softmax = nn.Softmax()
        self.sigmoid = nn.Sigmoid()

        # checkpoint_places = torch.load('/home/yjy/Code/PLACES365/resnet18_places365.pth.tar', map_location=lambda storage, loc: storage)
        checkpoint_faces = torch.load('models/PrivateTest_model_resnet18.t7')
        face_dict = self.face.state_dict()
        # places_dict = self.places.state_dict()
        # places_state_dict = {k[len('module.'):]: v for k, v in checkpoint_places['state_dict'].items() if k[len('module.'):] in places_dict}
        face_state_dict = {k: v for k, v in checkpoint_faces['net'].items() if k in face_dict}
        face_dict.update(face_state_dict)
        # places_dict.update(places_state_dict)

        self.face.load_state_dict(face_dict)

    #     self.weights_init()
    #
    # def weights_init(self):
    #     for m in self.modules():
    #
    #         if isinstance(m, nn.Linear):
    #             torch.nn.init.xavier_uniform_(m.weight.data)
    #             m.bias.data.fill_(0)

    def forward(self, x, face, rcnn, fmask):
        x1 = self.fcn(x)
        x = self.GAvgPool(x1)
        x = x.view(x.size(0), x.size(1))

        face, face1 = self.face(face)
        # face = face * fmask
        rcnn, _, alpha = self.lstm(rcnn)
        # rcnn = self.lstm(rcnn)


        # -------normalization---------#
        x_n = normalize(x)
        rcnn_n = normalize(rcnn)
        face_n = normalize(face)


        '''
        #-------adaptive weight---------#
        x_weight = self.classifiers_x(x)
        rcnn_weight = self.classifiers_rcnn(rcnn)
        face_weight = self.classifiers_face(face)

        weight = torch.cat([x_weight, rcnn_weight, face_weight], dim=1)
        weight = self.softmax(weight)

        # print(weight.max())

        x_weight = weight[:, 0]
        rcnn_weight = weight[:, 1]
        face_weight = weight[:, 2]

        x_weight = x_weight.view(x_weight.size(0), 1)
        rcnn_weight = rcnn_weight.view(rcnn_weight.size(0), 1)
        face_weight = face_weight.view(face_weight.size(0), 1)

        x = x * x_weight
        rcnn = rcnn * rcnn_weight
        face = face * face_weight
        '''

        #-------classifier8--------#
        features = torch.cat([x, rcnn, face], dim=1)
        # features = face
        emotion = self.classifier8(features)

        #-------8to2-------#s
        emotion = F.softmax(emotion, dim=1)

        positive = emotion[:, 0:4].sum(1)
        negative = emotion[:, 4:8].sum(1)

        positive = positive.view(positive.size(0), 1)
        negative = negative.view(negative.size(0), 1)

        sentiment = torch.cat([positive, negative], dim=1)

        # return emotion, sentiment, alpha, x, rcnn, face
        return emotion, sentiment


def normalize(x): # adding norm the network seems more stable

    e = 1e-5 * torch.ones(x.size(0), x.size(1)).cuda()
    x_max = x.max(1)[0].view(-1, 1).expand(x.size(0), x.size(1))
    x_min = x.min(1)[0].view(-1, 1).expand(x.size(0), x.size(1))
    x = (x - x_min) / (x_max - x_min + e)

    return x




