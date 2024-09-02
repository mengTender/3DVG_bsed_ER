import os
import torch
import torch.nn as nn
import torchvision
import dgl
from gcn import GCN
from vit import load_vit_base
from dgl import batch as dgl_batch
# from torch.utils.data

from torchvision import models


class PEB(nn.Module):
    def __init__(self, local_checkpoint_path='./pretrained_vit/vit_face.pth'):
        super(PEB, self).__init__()

        self.vit1 = load_vit_base(out_features=768, pretrained=False)
        # local_checkpoint = torch.load(local_checkpoint_path)
        # filtered_checkpoint = {k: v for k, v in local_checkpoint.items() if not k.startswith('head')}
        # self.vit1.load_state_dict(filtered_checkpoint, strict=False)

        self.vit2 = load_vit_base(out_features=768, pretrained=False)
        # self.vit2.heads = nn.Linear(768, 768)

        self.fc = nn.Linear(768 * 2, 512)

    def forward(self, face_img, pose_img):
        out1 = self.vit1(face_img)
        out2 = self.vit2(pose_img)
        combined_output = torch.cat((out1, out2), dim=1)
        combined_output = self.fc(combined_output)
        return combined_output


class CEB(nn.Module):
    def __init__(self):
        super(CEB, self).__init__()

        self.vit = models.vit_b_16(pretrained=False)
        self.vit.heads = nn.Linear(768, 512)
        self.gcn = GCN()
        self.fc = nn.Linear(1024, 512)

    def forward(self, context_img, graph):

        vit_out = self.vit(context_img)
        gcn_out = self.gcn(graph)
        combined_out = torch.cat((vit_out, gcn_out), dim=1)
        final_output = self.fc(combined_out)

        return final_output


class ModelForER(nn.Module):
    def __init__(self, out_feature):
        super(ModelForER, self).__init__()

        self.peb = PEB()
        self.ceb = CEB()

        self.fc = nn.Sequential(
            nn.Linear(in_features=1024, out_features=512),
            nn.ReLU(),
            nn.Linear(in_features=512, out_features=out_feature)
        )

    def forward(self, data):
        peb_out = self.peb(data['face'], data['pose'])
        ceb_out = self.ceb(data['bg_context'], data['3dvg'])
        combined_out = torch.cat((peb_out, ceb_out), dim=1)
        final_out = self.fc(combined_out)
        return peb_out, ceb_out, final_out


def collate_fn(batch_data):
    face_ls = []
    pose_ls = []
    bg_ls = []
    vg_ls = []
    for data in batch_data:
        face_ls.append(data['face'])
        pose_ls.append(data['pose'])
        bg_ls.append(data['bg_context'])
        vg_ls.append(data['3dvg'])
    batched_data = dict()
    batched_data['face'] = torch.stack(face_ls, dim=0)
    batched_data['pose'] = torch.stack(pose_ls, dim=0)
    batched_data['bg_context'] = torch.stack(bg_ls, dim=0)
    batched_data['3dvg'] = dgl_batch(vg_ls)
    return batched_data

