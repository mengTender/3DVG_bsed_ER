import os, sys
root_path = os.path.realpath(__file__).split('/evaluate/multipose_test.py')[0]
os.chdir(root_path)
sys.path.append(root_path)

from multiposenet_ext.network.posenet import poseNet
from multiposenet_ext.evaluate.tester import Tester

mask_model = poseNet(101)


def get_mask(img_path, output_path):
    backbone = 'resnet101'

    # Set Training parameters
    params = Tester.TestParams()
    params.subnet_name = 'detection_subnet'
    params.inp_size = 480
    params.coeff = 2
    params.in_thres = 0.21
    params.img_path = img_path
    params.testresult_dir = output_path
    params.testresult_write_image = True
    params.testresult_write_json = False
    params.ckpt = 'multiposenet_ext/ckpt/ckpt_baseline_resnet101.h5'


    for name, module in mask_model.named_children():
        for para in module.parameters():
            para.requires_grad = False

    tester = Tester(mask_model, params)
    tester.test()