import numpy as np
import torch
from tqdm import tqdm
from PIL import Image
import yaml, argparse, os, pickle
from utils import Namespace
from toto_benchmark.vision import load_model, load_transforms, preprocess_image

def precompute_embeddings(cfg, paths, data_path=None, from_files=False):
    device = 'cuda:0'
    model = load_model(cfg)
    model.to(device)
    model.eval()
    transforms = load_transforms(cfg)
    batch_size = 128
    print("Total number of paths: %i" % len(paths))
    path_images = []
    for img_dict in paths: 
        #breakpoint()-benchmark.org/
        img = Image.fromarray(img_dict["rgb"]) 
        img = img.convert("RGB")
        img = preprocess_image(img, transforms)  
        path_images.append(img.to(device)) 
        with torch.no_grad():
            img = img.unsqueeze(0).to(device)
            embedding = model(img).cpu().numpy()
            img_dict["rgb"] = embedding

    return paths

def precompute_embeddings_byol(cfg, paths, data_path):
    device = 'cuda:0'
    model = load_model(cfg)
    model.to(device)
    model = model.eval()
    byol_transforms = load_transforms(cfg)
    batch_size = 1
    print("Total number of paths : %i" % len(paths))
    for idx, path in tqdm(enumerate(paths)):
        path_images = []
        for t in range(path['observations'].shape[0]):
            img = Image.open(os.path.join(data_path, path['traj_id'], path['cam0c'][t]))  
            if cfg['data']['images']['crop']:
                img = img.crop((200, 0, 500, 400))
            img = preprocess_image(img, byol_transforms)
            path_images.append(img)
        embeddings = []
        path_len = len(path_images)
        with torch.no_grad():
            for b in range((path_len // batch_size + 1)):
                if b * batch_size < path_len:
                    chunk = torch.stack(path_images[b * batch_size:min(batch_size * (b + 1), path_len)])
                    chunk_embed = model(chunk.to(device))
                    embeddings.append(chunk_embed.to('cpu').data.numpy())
            embeddings = np.vstack(embeddings)
            assert embeddings.shape == (path_len, chunk_embed.shape[1])
        path['embeddings'] = embeddings.copy()
        path['observations'] = np.hstack([path['observations'], path['embeddings']])
    return paths
