import pickle
import numpy as np
from model.utils import get_augmentations
from model.trainer.helpers import (
    prepare_model
)
from model.image_translator.trainer import Translator_Trainer
from model.image_translator.utils import (
    get_recon, get_trans,
    loader_from_list, get_config
)
import argparse
import torch
from model.models.protonet import ProtoNet

sample_iters = 4000
AUGMENT = False
path = 'Animals-ConvNet-Pre/0.01_0.1_[75, 150, 300]/checkpoint.pth'
parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--dataset', type=str, default='Animals')    
parser.add_argument('--backbone_class', type=str, default='ConvNet', choices=['ConvNet', 'Res12'])
parser.add_argument('--query', type=int, default=1)    
parser.add_argument('--init_weights', type=str, default=path)
args = parser.parse_args()

config = get_config('animals.yaml')
loader = loader_from_list(
    root=config['data_folder_test'],
    file_list=config['data_list_test'],
    batch_size=1,
    new_size=140,
    height=128,
    width=128,
    crop=True,
    num_workers=1,
    return_paths=True,
    dataset=args.dataset)

train_loader = loader_from_list(
    root=config['data_folder_train'],
    file_list=config['data_list_train'],
    batch_size=config['pool_size'],
    new_size=140,
    height=128,
    width=128,
    crop=True,
    num_workers=4,
    dataset=args.dataset)
args.num_class = 119

model = ProtoNet(args)
loaded_dict = torch.load(path)['state_dict']
new_params = {}
keys = list(loaded_dict.keys())
for key in model.state_dict().keys():
    if key in keys:
        new_params[key] = loaded_dict[key]
    else:
        new_params[key] = model.state_dict()[key]
        print(f"Unexpected key {key}")

model.load_state_dict(new_params)

trainer = Translator_Trainer(config)
trainer.cuda()
trainer.load_ckpt('animals_gen.pt')
trainer.eval()

picker = Translator_Trainer(config)
picker.cuda()
picker.load_ckpt('animals_picker.pt')
picker = picker.model.gen
picker.eval()

model = model.cuda()
embeddings = []
all_embeddings = []
labels = []
AUGMENTATION_SIZE = 3

for i, data in enumerate(loader):
    if i%100 == 0:
        print(i)
    img = data[0].cuda()
    label = data[1].detach().cpu()
    path = data[-1][0]
    keep_idx = label < 20

    img = img[keep_idx]
    if len(img) == 0:
        continue
    label = label[keep_idx]
    if i >= sample_iters:
        break

    reconstructed_img = get_recon(config, path)
    embedding = model(reconstructed_img.unsqueeze(0), get_feature=True)
    if AUGMENT == True:
        class_expansions = get_trans(config, path, expansion_size=AUGMENTATION_SIZE)
        aug_embeddings = model(class_expansions, get_feature=True)
        embedding = torch.cat([embedding, aug_embeddings]).mean(0)
    
    all_embeddings.append(embedding.detach().cpu())
    labels.append(label)

all_embeddings = np.concatenate(all_embeddings).reshape(-1, 64)
print(all_embeddings.shape)
labels = np.concatenate(labels).reshape(-1, 1)
print(labels.shape)

with open('embeddings_overall_before.pkl', 'wb') as f:
    pickle.dump(all_embeddings, f)
with open('embeddings_label_overall_before.pkl', 'wb') as f:
    pickle.dump(labels, f)