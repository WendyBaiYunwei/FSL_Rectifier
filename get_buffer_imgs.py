# creates and saves the recon/translation images of the test imgs (3 copies)
# saved format:
# path_prefix/n02106030/n02106030_16184.JPEG_109_179_237_312_trans{1,2,3}.jpg
# path_prefix/n02106030/n02106030_16184.JPEG_109_179_237_312_recon.jpg

import numpy as np
from PIL import Image
import torch
from model.image_translator.utils import loader_from_list, get_config
from model.image_translator.trainer import Translator_Trainer

expansion_size = 3

config = get_config('./animals.yaml') # change to traffic.yaml for traffic
config['batch_size'] = 1

image_translator = Translator_Trainer(config)
image_translator.cuda()
if config['dataset'] == 'Animals':
    image_translator.load_ckpt('animals_gen.pt')
else:
    image_translator.load_ckpt('traffic_translator_gen.pt')
image_translator.eval()

picker = Translator_Trainer(config)
picker.cuda()
if config['dataset'] == 'Animals':
    picker.load_ckpt('animals_picker.pt')
    picker_loader = loader_from_list(
        root=config['data_folder_train'],
        file_list=config['data_list_train'],
        batch_size=config['pool_size'],
        new_size=140,
        height=128,
        width=128,
        crop=True,
        num_workers=4,
        dataset=config['dataset'])
else:
    picker.load_ckpt('traffic_picker.pt')
    picker_loader = loader_from_list(
        root=config['data_folder_test'],
        file_list=config['data_list_test'],
        batch_size=config['pool_size'],
        new_size=140,
        height=128,
        width=128,
        crop=True,
        num_workers=4,
        dataset=config['dataset'])

picker.eval()
picker = picker.model.gen

testloader = loader_from_list(
    root=config['data_folder_test'],
    file_list=config['data_list_test'],
    batch_size=config['batch_size'],
    new_size=140,
    height=128,
    width=128,
    crop=True,
    num_workers=4,
    return_paths=True,
    dataset='Animals') # pre-processing mode set to `Animals` to prevent CLAHE transformations for test samples

for i, data in enumerate(testloader):
    if i % 10 == 0:
        print(f'{i} / {len(testloader)}')
    original_img = data[0].cuda()
    label = data[1]
    paths = data[2]
    if config['dataset'] == 'Animals':
        imgs = image_translator.model.pick_animals(picker, original_img, picker_loader,\
             expansion_size=expansion_size, random=False, get_original=True, type='image_translator')
    else:
        imgs = image_translator.model.pick_traffic(picker, original_img, picker_loader,\
             expansion_size=expansion_size, random=False, get_original=True, type='image_translator')
    # save image
    for selected_i in range(expansion_size + 1):
        translation = imgs[selected_i]
        image = translation.detach().cpu().squeeze().numpy()
        image = np.transpose(image, (1, 2, 0))
        image = ((image + 1) * 0.5 * 255.0)
        output_img = Image.fromarray(np.uint8(image))
        
        original_path = '.'.join(paths[0].split('.')[:-1])
        if selected_i == 0:
            output_img.save(\
            f'{original_path}_recon.jpg', 'JPEG', quality=99)
        else:
            output_img.save(\
            f'{original_path}_trans{selected_i}.jpg', 'JPEG', quality=99)
