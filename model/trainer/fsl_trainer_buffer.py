import time
import os
import os.path as osp
import numpy as np

import torch
import torch.nn.functional as F

from model.trainer.base import Trainer
from model.trainer.helpers import (
    get_dataloader, prepare_model, prepare_optimizer,
)
from model.image_translator.utils import (
    get_recon, get_trans,
    get_train_loaders, get_config, get_dichomy_loader, loader_from_list
)
from model.utils import (
    pprint, ensure_path,
    Averager, Timer, count_acc, one_hot,
    compute_confidence_interval, get_augmentations
)
from collections import deque
from tqdm import tqdm

class FSLTrainer(Trainer):
    def __init__(self, args, config):
        super().__init__(args)
        self.config = config

        self.test_loader_fsl = get_dichomy_loader(
            episodes=config['max_iter'],
            root=config['data_folder_test'],
            file_list=config['data_list_test'],
            batch_size=config['batch_size'],
            new_size=140,
            height=128,
            width=128,
            crop=True,
            num_workers=4,
            return_paths=True,
            n_cls=config['way_size'],
            dataset='Animals')

        self.train_loader_image_translator = loader_from_list(
            root=config['data_folder_train'],
            file_list=config['data_list_train'],
            batch_size=config['pool_size'],
            new_size=140,
            height=128,
            width=128,
            crop=True,
            num_workers=4,
            dataset=config['dataset'])
        self.model, self.para_model = prepare_model(args)
        self.optimizer, self.lr_scheduler = prepare_optimizer(self.model, args)

    def prepare_label(self):
        args = self.args

        label = torch.arange(args.way, dtype=torch.int16).repeat(args.query)
        label_aux = torch.arange(args.way, dtype=torch.int8).repeat(args.shot + args.query)
        
        label = label.type(torch.LongTensor)
        label_aux = label_aux.type(torch.LongTensor)
        
        if torch.cuda.is_available():
            label = label.cuda()
            label_aux = label_aux.cuda()
            
        return label, label_aux
    
    def train(self):
        args = self.args
        loaded_dict = torch.load(args.model_path, map_location='cuda:0')
        if 'state_dict' in loaded_dict:
            loaded_dict = loaded_dict['state_dict']
        else:
            loaded_dict = loaded_dict['params']
        new_params = {}
        keys = list(loaded_dict.keys())

        for key in self.model.state_dict().keys():
            if key in keys:
                new_params[key] = loaded_dict[key]
            else:
                new_params[key] = self.model.state_dict()[key]
        self.model.load_state_dict(new_params)
        self.model.train()
        
        if self.args.fix_BN:
            self.model.encoder.eval()
        
        # start FSL training
        label, label_aux = self.prepare_label()
        
        tl1 = Averager()
        tl2 = Averager()
        ta = Averager()

        for it, batch in enumerate(self.train_loader):
            self.train_step += 1

            if self.train_step > args.max_epoch * args.episodes_per_epoch:
                break
            if torch.cuda.is_available():
                data, gt_label = [_.cuda() for _ in batch]
            else:
                data, gt_label = batch[0], batch[1]
            
            data_tm = time.time()

            # get saved centers
            logits, reg_logits = self.para_model(data)
            if reg_logits is not None:
                loss = F.cross_entropy(logits, label)
                total_loss = loss + args.balance * F.cross_entropy(reg_logits, label_aux)
            else:
                loss = F.cross_entropy(logits, label)
                total_loss = F.cross_entropy(logits, label)
                
            tl2.add(loss)
            forward_tm = time.time()
            self.ft.add(forward_tm - data_tm)
            acc = count_acc(logits, label)

            tl1.add(total_loss.item())
            ta.add(acc)

            self.optimizer.zero_grad()
            total_loss.backward()
            backward_tm = time.time()
            self.bt.add(backward_tm - forward_tm)

            self.optimizer.step()
            optimizer_tm = time.time()
            self.ot.add(optimizer_tm - backward_tm)    

        if not osp.exists(args.save_path):
            os.mkdir(args.save_path)

        torch.save(self.trlog, osp.join(args.save_path, 'trlog'))
        self.save_model('epoch-last')

    def evaluate(self, data_loader):
        # restore model args
        args = self.args
        # evaluation mode
        self.model.eval()
        record = np.zeros((args.num_eval_episodes, 2)) # loss and acc
        label = torch.arange(args.eval_way, dtype=torch.int16).repeat(args.eval_query)
        label = label.type(torch.LongTensor)
        if torch.cuda.is_available():
            label = label.cuda()
        print('best epoch {}, best val acc={:.4f} + {:.4f}'.format(
                self.trlog['max_acc_epoch'],
                self.trlog['max_acc'],
                self.trlog['max_acc_interval']))
        with torch.no_grad():
            for i, batch in enumerate(data_loader, 1):
                if torch.cuda.is_available():
                    data, _ = [_.cuda() for _ in batch]
                else:
                    data = batch[0]

                logits = self.model(data)
                loss = F.cross_entropy(logits, label)
                acc = count_acc(logits, label)
                record[i-1, 0] = loss.item()
                record[i-1, 1] = acc
                
        assert(i == record.shape[0])
        vl, _ = compute_confidence_interval(record[:,0])
        va, vap = compute_confidence_interval(record[:,1])
        
        # train mode
        self.model.train()
        if self.args.fix_BN:
            self.model.encoder.eval()

        return vl, va, vap

    def evaluate_test(self):
        # restore model args
        args = self.args
        from model.image_translator.trainer import Translator_Trainer
        trainer = Translator_Trainer(self.config)
        trainer.cuda()
        trainer.load_ckpt('animals_gen.pt')
        trainer.eval()
        self.trainer = trainer
        picker = Translator_Trainer(self.config)
        picker.cuda()
        picker.load_ckpt('animals_picker.pt')
        picker.eval()
        picker = picker.model.gen
        self.picker = picker
        loaded_dict = torch.load(args.model_path)
        if 'state_dict' in loaded_dict:
            loaded_dict = loaded_dict['state_dict']
        else:
            loaded_dict = loaded_dict['params']
        new_params = {}
        keys = list(loaded_dict.keys())

        for key in self.model.state_dict().keys():
            if key in keys:
                new_params[key] = loaded_dict[key]
            else:
                new_params[key] = self.model.state_dict()[key]

        self.model.load_state_dict(new_params) 
        self.model.eval()
        baseline = np.zeros((args.num_eval_episodes, 2)) 
        i2name = {0: 'image_translator'}

        expansion_res = []
        for i in i2name:
            expansion_res.append(np.zeros((args.num_eval_episodes, 2))) # loss and acc
        label = torch.arange(args.eval_way, dtype=torch.int16).repeat(args.eval_query)
        label = label.type(torch.LongTensor)
        if torch.cuda.is_available():
            label = label.cuda()

        qry_expansion = args.qry_expansion
        spt_expansion = args.spt_expansion
        old_shot = args.eval_shot
        old_qry = args.eval_query
        with torch.no_grad():
            for i, batch in enumerate(self.test_loader_fsl):
                if i % 100 == 0:
                    print(i)
                if i >= args.num_eval_episodes:
                    break
                if torch.cuda.is_available():
                    data, _ = [_.cuda() for _ in batch[:-1]]
                else:
                    data = batch[0]
                paths = batch[-1]
                new_data = torch.empty(data.shape).cuda()

                for img_i in range(len(new_data)):
                    img_name = paths[img_i]
                    new_data[img_i] = get_recon(self.config, img_name)
                logits = self.model(new_data)
                loss = F.cross_entropy(logits, label)
                acc = count_acc(logits, label)
                baseline[i-1, 0] = loss.item()
                baseline[i-1, 1] = acc

                for type_i in i2name:
                    name = i2name[type_i]

                    original_spt = data[:5, :, :, :]
                    reconstructed_spt = new_data[:5, :, :, :]

                    img_names = paths[:5]
                    if spt_expansion == 0 and self.args.add_transform == None:
                        combined_spt = reconstructed_spt
                    elif self.args.add_transform and \
                        name in ['image_translator', 'mix-up', 'random-mix-up', 'random-image_translator']: # use original data:
                        expanded_spt = self.get_class_expansion(picker, original_spt,\
                             spt_expansion, img_names = img_names, type=name)
                        additional_spt = self.get_class_expansion(picker,\
                             reconstructed_spt, spt_expansion,\
                             img_names = img_names, type=self.args.add_transform)  
                        combined_spt = torch.cat([reconstructed_spt, expanded_spt, additional_spt], dim=0)
                    elif self.args.add_transform:
                        expanded_spt = self.get_class_expansion(picker, reconstructed_spt,\
                         spt_expansion, type=name) 
                        additional_spt = self.get_class_expansion(picker, reconstructed_spt,\
                         spt_expansion, type=name)  
                        combined_spt = torch.cat([reconstructed_spt, expanded_spt, additional_spt], dim=0)                  
                    elif name in ['image_translator', 'mix-up', 'random-mix-up', 'random-image_translator']:
                        expanded_spt = self.get_class_expansion(picker, original_spt,\
                         spt_expansion, img_names = img_names, type=name)
                        combined_spt = torch.cat([reconstructed_spt, expanded_spt], dim=0)
                    else:
                        expanded_spt = self.get_class_expansion(picker, reconstructed_spt,\
                         spt_expansion, img_names = img_names, type=name)
                        combined_spt = torch.cat([reconstructed_spt, expanded_spt], dim=0)

                    original_qry = new_data[5:, :, :, :]
                    new_qries = torch.empty(old_qry, 5 * qry_expansion, data.shape[1], data.shape[2],\
                     data.shape[3]).cuda()

                    k = 0
                    if name in ['image_translator', 'mix-up', 'random-mix-up', 'random-image_translator']: # use original data
                        for class_chunk_i in range(5, len(data), 5):
                            class_chunk = data[class_chunk_i:class_chunk_i+5]
                            new_qries[k] = self.get_class_expansion(picker, class_chunk, qry_expansion,\
                            img_names = img_names, type=name)
                            k += 1
                    else:# use restructured data
                        for class_chunk_i in range(5, len(new_data), 5):
                            class_chunk = new_data[class_chunk_i:class_chunk_i+5]
                            new_qries[k] = self.get_class_expansion(picker, class_chunk, qry_expansion, type=name)
                            k += 1
                    assert k == old_qry
                    new_qries = new_qries.flatten(end_dim=1)
                    expanded_data = torch.cat([combined_spt, original_qry, new_qries], dim=0)
                    if self.args.add_transform:
                        logits = self.model(expanded_data, qry_expansion=qry_expansion, spt_expansion=spt_expansion*2)
                    else:
                        logits = self.model(expanded_data, qry_expansion=qry_expansion, spt_expansion=spt_expansion)
                    loss = F.cross_entropy(logits, label)
                    acc = count_acc(logits, label)
                    expansion_res[type_i][i-1, 0] = loss.item()
                    expansion_res[type_i][i-1, 1] = acc

        assert(i == baseline.shape[0])
        vl, _ = compute_confidence_interval(baseline[:,0])
        va, vap = compute_confidence_interval(baseline[:,1])
        
        result_str = ''
        result_str += 'Baseline Test acc={:.4f} + {:.4f}\n'.format(va, vap)

        for type_i in i2name:
            name = i2name[type_i]
            vl, _ = compute_confidence_interval(expansion_res[type_i][:,0])
            va, vap = compute_confidence_interval(expansion_res[type_i][:,1])
            
            result_str += f'{name} Test acc={va} + {vap}\n'

        with open(f'./outputs/{args.model_class}-{args.backbone_class}-{args.dataset}-{args.use_euclidean}-{args.\
            add_transform}-{args.spt_expansion}-{args.qry_expansion}-record.txt', 'w') as file:
            file.write(result_str)
        return vl, va, vap
    
    # input 01234, return 012340123401234
    # 0: 'oracle', 1: 'mix_up', 2: 'affine', 3: 'color', 4: 'crops_flip_scale', 5: 'image_translator'
    def get_class_expansion(self, picker, data, expansion, type='image_translator', img_names=''):
        expanded = torch.empty(5, expansion, data.shape[1], data.shape[2], data.shape[3]).cuda()
        for class_i in range(5):
            img = data[class_i].unsqueeze(0)
            if type == 'image_translator' or type == 'mix-up':
                class_expansions = get_trans(self.config, img_names[class_i], expansion_size=expansion)
            elif type == 'random-mix-up' or type == 'random-image_translator':
                class_expansions = self.trainer.model.pick_animals(self.picker, img, self.train_loader_image_translator, \
                        expansion_size=expansion, random=True, get_img=False, get_original=False, type=type)
            else:
                class_expansions = get_augmentations(img, expansion, type, get_img=False)
            expanded[class_i] = class_expansions
        expanded = expanded.swapaxes(0, 1).flatten(end_dim=1)
        return expanded
        
    def final_record(self):
        with open(osp.join(self.args.save_path, '{}+{}'.format(self.trlog['test_acc'], self.trlog['test_acc_interval'])), 'w') as f:
            f.write('best epoch {}, best val acc={:.4f} + {:.4f}\n'.format(
                self.trlog['max_acc_epoch'],
                self.trlog['max_acc'],
                self.trlog['max_acc_interval']))
            f.write('Test acc={:.4f} + {:.4f}\n'.format(
                self.trlog['test_acc'],
                self.trlog['test_acc_interval']))            