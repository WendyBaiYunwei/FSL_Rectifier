
import os
import yaml
import time

import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import torchvision.utils as vutils

from model.image_translator.data import ImageLabelFilelist, default_loader
import numpy as np


def update_average(model_tgt, model_src, beta=0.999):
    with torch.no_grad():
        param_dict_src = dict(model_src.named_parameters())
        for p_name, p_tgt in model_tgt.named_parameters():
            p_src = param_dict_src[p_name]
            assert(p_src is not p_tgt)
            p_tgt.copy_(beta*p_tgt + (1. - beta)*p_src)


def loader_from_list(
        root,
        file_list,
        batch_size,
        new_size=None,
        height=128,
        width=128,
        crop=True,
        num_workers=4,
        shuffle=True,
        center_crop=False,
        return_paths=False,
        drop_last=True,
        dataset='Animals'):
    transform = get_transform(new_size, height, width)

    dataset = ImageLabelFilelist(root,
                                file_list,
                                transform,
                                dataset=dataset,
                                return_paths=return_paths)
    loader = DataLoader(dataset,
                        batch_size,
                        shuffle=shuffle,
                        drop_last=drop_last,
                        num_workers=num_workers)
    return loader


def get_evaluation_loaders(conf, shuffle_content=False):
    batch_size = conf['batch_size']
    num_workers = conf['num_workers']
    new_size = conf['new_size']
    width = conf['crop_image_width']
    height = conf['crop_image_height']
    content_loader = loader_from_list(
            root=conf['data_folder_train'],
            file_list=conf['data_list_train'],
            batch_size=batch_size,
            new_size=new_size,
            height=height,
            width=width,
            crop=True,
            num_workers=num_workers,
            shuffle=shuffle_content,
            center_crop=True,
            return_paths=True,
            drop_last=False,
            dataset=conf['dataset'])

    class_loader = loader_from_list(
            root=conf['data_folder_test'],
            file_list=conf['data_list_test'],
            batch_size=batch_size * conf['k_shot'],
            new_size=new_size,
            height=height,
            width=width,
            crop=True,
            num_workers=1,
            shuffle=False,
            center_crop=True,
            return_paths=True,
            drop_last=False,
            dataset=conf['dataset'])
    return content_loader, class_loader

# returns batch_size x num_cls x img_dim 
def get_dichomy_loader(
        episodes,
        root,
        file_list,
        batch_size,
        new_size=None,
        height=128,
        width=128,
        crop=True,
        num_workers=4,
        shuffle=True,
        center_crop=False,
        return_paths=False,
        drop_last=True,
        n_cls=2,
        dataset='Animals'):

    transform = get_transform(new_size, height, width)
    dataset = ImageLabelFilelist(root,
                                 file_list,
                                 transform,
                                 return_paths=return_paths,
                                 dataset=dataset)

    train_sampler = CategoriesSampler(dataset.labels,
                                    n_batch=episodes,
                                    n_cls=n_cls,
                                    n_per=batch_size)

    loader = DataLoader(dataset=dataset,
                        num_workers=num_workers,
                        batch_sampler=train_sampler,
                        pin_memory=True)

    return loader

class CategoriesSampler():
    def __init__(self, label, n_batch, n_cls, n_per):
        self.n_batch = n_batch
        self.n_cls = n_cls
        self.n_per = n_per

        label = np.array(label)
        self.m_ind = []
        for i in range(max(label) + 1):
            ind = np.argwhere(label == i).reshape(-1)
            ind = torch.from_numpy(ind)
            self.m_ind.append(ind)

    def __len__(self):
        return self.n_batch

    def __iter__(self):
        for i_batch in range(self.n_batch):
            batch = []
            classes = torch.randperm(len(self.m_ind))[:self.n_cls]
            for c in classes:
                l = self.m_ind[c]
                pos = torch.randperm(len(l))[:self.n_per]
                batch.append(l[pos])
            batch = torch.stack(batch).t().reshape(-1)
            yield batch

def get_train_loaders(conf):
    batch_size = conf['batch_size']
    num_workers = conf['num_workers']
    new_size = conf['new_size']
    width = conf['crop_image_width']
    height = conf['crop_image_height']
    train_content_loader = loader_from_list(
            root=conf['data_folder_train'],
            file_list=conf['data_list_train'],
            batch_size=batch_size,
            new_size=new_size,
            height=height,
            width=width,
            crop=True,
            num_workers=num_workers,
            dataset=conf['dataset'])
    train_class_loader = loader_from_list(
            root=conf['data_folder_train'],
            file_list=conf['data_list_train'],
            batch_size=batch_size,
            new_size=new_size,
            height=height,
            width=width,
            crop=True,
            num_workers=num_workers,
            dataset=conf['dataset'])
    test_content_loader = loader_from_list(
            root=conf['data_folder_test'],
            file_list=conf['data_list_test'],
            batch_size=batch_size,
            new_size=new_size,
            height=height,
            width=width,
            crop=True,
            num_workers=1,
            dataset=conf['dataset'])
    test_class_loader = loader_from_list(
            root=conf['data_folder_test'],
            file_list=conf['data_list_test'],
            batch_size=batch_size,
            new_size=new_size,
            height=height,
            width=width,
            crop=True,
            num_workers=1,
            dataset=conf['dataset'])

    return (train_content_loader, train_class_loader, test_content_loader,
            test_class_loader)

def get_dichomy_loaders(conf):
    num_workers = conf['num_workers']
    new_size = conf['new_size']
    width = conf['crop_image_width']
    height = conf['crop_image_height']
    
    train_loader = get_dichomy_loader(
            episodes=conf['max_iter'],
            root=conf['data_folder_train'],
            file_list=conf['data_list_train'],
            batch_size=conf['batch_size'],
            new_size=new_size,
            height=height,
            width=width,
            crop=True,
            num_workers=num_workers,
            n_cls=2,
            dataset=conf['dataset'])

    test_loader = get_dichomy_loader(
            episodes=conf['max_iter'],
            root=conf['data_folder_test'],
            file_list=conf['data_list_test'],
            batch_size=conf['batch_size'],
            new_size=new_size,
            height=height,
            width=width,
            crop=True,
            num_workers=num_workers,
            n_cls=2,
            dataset=conf['dataset'])
    
    test_loader_fsl = get_dichomy_loader(
            episodes=conf['max_iter'],
            root=conf['data_folder_test'],
            file_list=conf['data_list_test'],
            batch_size=conf['eval_shot'] + conf['eval_query'],
            new_size=new_size,
            height=height,
            width=width,
            crop=True,
            num_workers=num_workers,
            n_cls=conf['way_size'],
            return_paths=True,
            dataset=conf['dataset'])

    train_loader_fsl = get_dichomy_loader(
            episodes=conf['max_iter'],
            root=conf['data_folder_test'],
            file_list=conf['data_list_test'],
            batch_size=conf['eval_shot'] + conf['eval_query'],
            new_size=new_size,
            height=height,
            width=width,
            crop=True,
            num_workers=num_workers,
            n_cls=conf['way_size'],
            dataset=conf['dataset'])
    
    return train_loader, test_loader, test_loader_fsl, train_loader_fsl

def get_pretrain_loaders(conf):
    batch_size = conf['batch_size']
    num_workers = conf['num_workers']
    new_size = conf['new_size']
    width = conf['crop_image_width']
    height = conf['crop_image_height']

    train_loader = loader_from_list(
            root=conf['data_folder_train'],
            file_list=conf['data_list_train'],
            batch_size=batch_size,
            new_size=new_size,
            height=height,
            width=width,
            crop=True,
            shuffle=True,
            num_workers=num_workers,
            dataset=conf['dataset'])

    return train_loader

def get_transform(new_size, height, width):
    transform_list = [transforms.Resize(new_size), transforms.CenterCrop((height, width)), \
            transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]

    transform = transforms.Compose(transform_list)
    return transform


def get_recon(conf, img_name):
    img_name = '.'.join(img_name.split('.')[:-1])
    name = img_name + f'_recon.jpg'
    image = default_loader(name)
    transform = get_transform(conf['new_size'],\
         conf['crop_image_height'], conf['crop_image_width'])
    image = transform(image).cuda()
    return image # 1,3,128,128

def get_trans(conf, img_name, expansion_size):
    images = torch.empty(expansion_size, 3, conf['crop_image_height'],\
        conf['crop_image_width']).cuda()
    img_name = '.'.join(img_name.split('.')[:-1])
    transform = get_transform(conf['new_size'],\
        conf['crop_image_height'], conf['crop_image_width'])
    for i in range(1, expansion_size + 1):
        name = img_name + f'_trans{i}.jpg'
        image = default_loader(name)
        image = transform(image)
        images[i-1] = image
    return images

def get_config(config):
    with open(config, 'r') as stream:
        return yaml.load(stream, Loader=yaml.FullLoader)


def make_result_folders(output_directory):
    image_directory = os.path.join(output_directory, 'images')
    if not os.path.exists(image_directory):
        print("Creating directory: {}".format(image_directory))
        os.makedirs(image_directory)
    checkpoint_directory = os.path.join(output_directory, 'checkpoints')
    if not os.path.exists(checkpoint_directory):
        print("Creating directory: {}".format(checkpoint_directory))
        os.makedirs(checkpoint_directory)
    return checkpoint_directory, image_directory


def __write_images(im_outs, dis_img_n, file_name):
    im_outs = [images.expand(-1, 3, -1, -1) for images in im_outs]
    image_tensor = torch.cat([images[:dis_img_n] for images in im_outs], 0)
    image_grid = vutils.make_grid(image_tensor.data,
                                  nrow=dis_img_n, padding=0, normalize=True)
    vutils.save_image(image_grid, file_name, nrow=1)


def write_1images(image_outputs, image_directory, postfix):
    display_image_num = image_outputs[0].size(0)
    __write_images(image_outputs, display_image_num,
                   '%s/gen_%s.jpg' % (image_directory, postfix))


def _write_row(html_file, it, fn, all_size):
    html_file.write("<h3>iteration [%d] (%s)</h3>" % (it, fn.split('/')[-1]))
    html_file.write("""
        <p><a href="%s">
          <img src="%s" style="width:%dpx">
        </a><br>
        <p>
        """ % (fn, fn, all_size))
    return


def write_html(filename, it, img_save_it, img_dir, all_size=1536):
    html_file = open(filename, "w")
    html_file.write('''
    <!DOCTYPE html>
    <html>
    <head>
      <title>Experiment name = %s</title>
      <meta http-equiv="refresh" content="30">
    </head>
    <body>
    ''' % os.path.basename(filename))
    html_file.write("<h3>current</h3>")
    _write_row(html_file, it, '%s/gen_train_current.jpg' % img_dir, all_size)
    for j in range(it, img_save_it - 1, -1):
        _write_row(html_file, j, '%s/gen_train_%08d.jpg' % (img_dir, j),
                   all_size)
    html_file.write("</body></html>")
    html_file.close()


def write_loss(iterations, trainer, train_writer):
    members = [attr for attr in dir(trainer)
               if ((not callable(getattr(trainer, attr))
                    and not attr.startswith("__"))
                   and ('loss' in attr
                        or 'grad' in attr
                        or 'nwd' in attr
                        or 'accuracy' in attr))]
    for m in members:
        train_writer.add_scalar(m, getattr(trainer, m), iterations + 1)


class Timer:
    def __init__(self, msg):
        self.msg = msg
        self.start_time = None

    def __enter__(self):
        self.start_time = time.time()

    def __exit__(self, exc_type, exc_value, exc_tb):
        print(self.msg % (time.time() - self.start_time))

def kl_divergence(p, q):
    num_classes = p.shape[-1]
    p += torch.abs(torch.min(p)) + 1e-3
    norm_factor = torch.sum(p, dim=1).unsqueeze(1)
    p /= norm_factor.repeat(1, num_classes)
    q += torch.abs(torch.min(q)) + 1e-3
    norm_factor = torch.sum(q, dim=1).unsqueeze(1)
    q /= norm_factor.repeat(1, num_classes)
    kl = (p * (torch.log(p) - torch.log(q))).sum()
    return kl