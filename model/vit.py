# load pre-trained Vision Transformer
import timm
import torch
import torch.nn as nn


def load_vit_base(out_features=1024, pretrained=True):
    model = timm.create_model('vit_base_patch16_224', pretrained=pretrained)
    model.head = nn.Linear(model.head.in_features, out_features)
    return model


def load_vit_large(out_features=1024, pretrained=True):
    model = timm.create_model('vit_large_patch16_224', pretrained=pretrained)
    model.head = nn.Linear(model.head.in_features, out_features)
    return model


def load_vit_huge(out_features=1024, pretrained=True):
    model = timm.create_model('vit_huge_patch14_224', pretrained=pretrained)
    model.head = nn.Linear(model.head.in_features, out_features)
    return model


# if __name__ == "__main__":
#     vit_base_model = load_vit_base()
#     print("Loaded ViT Base model with output features dimension of 1024:")
#     print(vit_base_model)
#
#     vit_large_model = load_vit_large()
#     print("\nLoaded ViT Large model with output features dimension of 1024:")
#     print(vit_large_model)
#
#     vit_huge_model = load_vit_huge()
#     print("\nLoaded ViT Huge model with output features dimension of 1024:")
#     print(vit_huge_model)