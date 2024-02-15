import numpy as np, torch, torch.nn as nn, torchvision.transforms as T, os, sys
from pathlib import Path
from PIL import Image
import torchvision.models as models
from torchvision.transforms import InterpolationMode
from torch.nn.modules.linear import Identity

CHECKPOINT_DIR = "/home/jess/toto_benchmark/assets"

_resnet_transforms = T.Compose([
                        T.Resize(256),
                        T.CenterCrop(224),
                        T.ToTensor(),
                        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                        ])

_r3m_transforms = T.Compose([
                        T.Resize(256),
                        T.CenterCrop(224),
                        T.ToTensor(),  # this divides by 255
                        T.Normalize(mean=[0.0, 0.0, 0.0], std=[1.0/255, 1.0/255, 1.0/255]), # this will scale bact to [0-255]
                    ])


MODEL_LIST = ['resnet50', 'resnet50_rand', 'clip_vit', 'clip_rn50',
              'moco_conv5', 'moco_conv4', 'moco_conv3',
              'moco_croponly_conv5', 'moco_croponly_conv4', 'moco_croponly_conv3',
              'r3m', 'moco_conv5_robocloud'
             ]


def load_pvr_transforms(embedding_name, *args, **kwargs):
    
    assert embedding_name in MODEL_LIST
    
    # ============================================================
    # ResNet50
    # ============================================================
    if embedding_name == 'resnet50':
        # ResNet50 pretrained on ImageNet
        embedding_dim, transforms = 2048, _resnet_transforms
    elif embedding_name == 'resnet50_rand':
        # Randomly initialized ResNet50 features
        embedding_dim, transforms = 2048, _resnet_transforms
    # ============================================================
    # MoCo (Aug+)
    # ============================================================
    elif embedding_name == 'moco_conv3':
        embedding_dim = 2156
        transforms = _resnet_transforms
    elif embedding_name == 'moco_conv4':
        model, embedding_dim = moco_conv4_compression_model(CHECKPOINT_DIR + '/moco_v2_conv4.pth.tar')
        transforms = _resnet_transforms
    elif embedding_name == 'moco_conv5':
        embedding_dim = 2048
        transforms = _resnet_transforms
    elif embedding_name == 'moco_conv5_robocloud':
        embedding_dim = 2048
        transforms = _resnet_transforms
    # ============================================================
    # MoCo (croponly)
    # ============================================================
    elif embedding_name == 'moco_croponly_conv3':
        model, embedding_dim = moco_conv3_compression_model(CHECKPOINT_DIR + '/moco_croponly_conv3.pth')
        transforms = _resnet_transforms
    elif embedding_name == 'moco_croponly_conv4':
        model, embedding_dim = moco_conv4_compression_model(CHECKPOINT_DIR + '/moco_croponly_conv4.pth')
        transforms = _resnet_transforms
    elif embedding_name == 'moco_croponly_conv5':
        model, embedding_dim = moco_conv5_model(CHECKPOINT_DIR + '/moco_croponly.pth')
        transforms = _resnet_transforms
    # ============================================================
    # R3M
    # ============================================================
    elif embedding_name == 'r3m':
        from r3m import load_r3m
        embedding_dim = 2048
        transforms = _r3m_transforms
    else:
        print("Model not implemented.")
        raise NotImplementedError
    # model = model.eval()
    return embedding_dim, transforms

    
def load_pvr_model(embedding_name, *args, **kwargs):
    assert embedding_name in MODEL_LIST
    
    # ============================================================
    # ResNet50
    # ============================================================
    if embedding_name == 'resnet50':
        # ResNet50 pretrained on ImageNet
        model = models.resnet50(pretrained=True, progress=False)
        model.fc = Identity()
        embedding_dim, transforms = 2048, _resnet_transforms
    elif embedding_name == 'resnet50_rand':
        # Randomly initialized ResNet50 features
        model = models.resnet50(pretrained=False, progress=False)
        model.fc = Identity()
        embedding_dim, transforms = 2048, _resnet_transforms
    # ============================================================
    # MoCo (Aug+)
    # ============================================================
    elif embedding_name == 'moco_conv3':
        model, embedding_dim = moco_conv3_compression_model(CHECKPOINT_DIR + '/moco_v2_conv3.pth.tar')
        transforms = _resnet_transforms
    elif embedding_name == 'moco_conv4':
        model, embedding_dim = moco_conv4_compression_model(CHECKPOINT_DIR + '/moco_v2_conv4.pth.tar')
        transforms = _resnet_transforms
    elif embedding_name == 'moco_conv5':
        model, embedding_dim = moco_conv5_model(CHECKPOINT_DIR + '/moco_v2_800ep_pretrain.pth.tar')
        transforms = _resnet_transforms
    elif embedding_name == 'moco_conv5_robocloud':
        model, embedding_dim = moco_conv5_model(CHECKPOINT_DIR + '/moco_conv5_robocloud.pth')
        transforms = _resnet_transforms
        print(f"loaded from {CHECKPOINT_DIR + '/moco_conv5_robocloud.pth'}")
    # ============================================================
    # MoCo (croponly)
    # ============================================================
    elif embedding_name == 'moco_croponly_conv3':
        model, embedding_dim = moco_conv3_compression_model(CHECKPOINT_DIR + '/moco_croponly_conv3.pth')
        transforms = _resnet_transforms
    elif embedding_name == 'moco_croponly_conv4':
        model, embedding_dim = moco_conv4_compression_model(CHECKPOINT_DIR + '/moco_croponly_conv4.pth')
        transforms = _resnet_transforms
    elif embedding_name == 'moco_croponly_conv5':
        model, embedding_dim = moco_conv5_model(CHECKPOINT_DIR + '/moco_croponly.pth')
        transforms = _resnet_transforms
    # ============================================================
    # R3M
    # ============================================================
    elif embedding_name == 'r3m':
        from r3m import load_r3m
        model = load_r3m("resnet50")
        model = model.module.eval()
        model = model.to('cpu')
        embedding_dim = 2048
        transforms = _r3m_transforms
    else:
        print("Model not implemented.")
        raise NotImplementedError
    model = model.eval()
    return model, embedding_dim, transforms


def load_pretrained_policy(embedding_name, policy_path):
    print(embedding_name)
    from pvr_model_training import VisuoMotorPolicy
    base_model, embedding_dim, transforms = load_pvr_model(embedding_name)
    control_policy = torch.load(policy_path, map_location='cpu')
    policy = VisuoMotorPolicy(base_model=base_model, policy=control_policy)
    return policy, transforms


def moco_conv5_model(checkpoint_path):
    print("moco_conv5 loaded")
    model = models.resnet50(pretrained=False, progress=False)
    checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
    # rename moco pre-trained keys
    state_dict = checkpoint['state_dict']
    for k in list(state_dict.keys()):
        # retain only encoder_q up to before the embedding layer
        if k.startswith('module.encoder_q'
                        ) and not k.startswith('module.encoder_q.fc'):
            # remove prefix
            state_dict[k[len("module.encoder_q."):]] = state_dict[k]
        # delete renamed or unused k
        del state_dict[k]
    msg = model.load_state_dict(state_dict, strict=False)
    assert set(msg.missing_keys) == {"fc.weight", "fc.bias"}
    model.fc = Identity()
    return model, 2048


def moco_conv4_compression_model(checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
    state_dict = checkpoint['state_dict']
    # construct the compressed model
    model = models.resnet.resnet50(pretrained=False, progress=False)
    downsample = nn.Sequential(
                    nn.Conv2d(2048,
                    42,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    groups=1,
                    dilation=1), model._norm_layer(42))
    model.layer4 = nn.Sequential(
                    model.layer4,
                    models.resnet.BasicBlock(2048,
                        42,
                        stride=1,
                        norm_layer=model._norm_layer,
                        downsample=downsample))
    # Remove the avgpool layer
    model.avgpool = nn.Sequential()
    model.fc = nn.Sequential()

    for k in list(state_dict.keys()):
        # retain only encoder_q up to before the embedding layer
        if k.startswith('module.encoder_q'
                        ) and not k.startswith('module.encoder_q.fc'):
            # remove prefix
            state_dict[k[len("module.encoder_q."):]] = state_dict[k]
        # delete renamed or unused k
        del state_dict[k]

    msg = model.load_state_dict(state_dict, strict=False)
    assert all(['fc.' in n or 'layer4.2' in n  for n in msg.unexpected_keys])
    assert len(msg.missing_keys)==0
    # manually computed the embedding dimension to be 2058
    return model, 2058


def moco_conv3_compression_model(checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
    state_dict = checkpoint['state_dict']
    # construct the compressed model
    model = models.resnet.resnet50(pretrained=False, progress=False)
    downsample1 = nn.Sequential(
        nn.Conv2d(1024,
                  11,
                  kernel_size=3,
                  stride=1,
                  padding=1,
                  groups=1,
                  dilation=1), model._norm_layer(11))

    model.layer3 = nn.Sequential(
        model.layer3,
        models.resnet.BasicBlock(1024,
                                 11,
                                 stride=1,
                                 norm_layer=model._norm_layer,
                                 downsample=downsample1)
    )

    # Remove the avgpool layer
    model.layer4 = nn.Sequential()
    model.avgpool = nn.Sequential()
    model.fc = nn.Sequential()

    for k in list(state_dict.keys()):
        # retain only encoder_q up to before the embedding layer
        if k.startswith('module.encoder_q'
                        ) and not k.startswith('module.encoder_q.fc'):
            # remove prefix
            state_dict[k[len("module.encoder_q."):]] = state_dict[k]
        # delete renamed or unused k
        del state_dict[k]

    msg = model.load_state_dict(state_dict, strict=False)
    assert all(['fc.' in n or 'layer4.' in n or 'layer3.2' in n for n in msg.unexpected_keys])
    assert len(msg.missing_keys)==0
    # manually computed the embedding dimension to be 2156
    return model, 2156
