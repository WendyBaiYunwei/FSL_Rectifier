import torch
import torch.backends.cudnn as cudnn

import os
import sys
import argparse
import shutil

from model.image_translator.utils import write_loss, write_html,\
 write_1images, Timer, get_dichomy_loaders, get_config, get_train_loaders, make_result_folders
from model.image_translator.trainer import Translator_Trainer

cudnn.benchmark = True

parser = argparse.ArgumentParser()
parser.add_argument('--config',
                    type=str,
                    default='animals.yaml',
                    help='configuration file for training and testing')
parser.add_argument('--output_path',
                    type=str,
                    default='.',
                    help="outputs path")
parser.add_argument('--multigpus',
                    action="store_true")
parser.add_argument('--batch_size',
                    type=int,
                    default=0)
parser.add_argument('--test_batch_size',
                    type=int,
                    default=4)
opts = parser.parse_args()

config = get_config(opts.config)
max_iter = config['max_iter'] + 1000

trainer = Translator_Trainer(config)
trainer.cuda()

config['gpus'] = 1
loaders = get_train_loaders(config)
train_loader = loaders[0]
test_loader = loaders[2]

model_name = os.path.splitext(os.path.basename(opts.config))[0]

if config['dataset'] == 'Traffic':
    output_directory = os.path.join(opts.output_path + "/outputs/traffic_picker", model_name)
elif config['dataset'] == 'Animals':
    output_directory = os.path.join(opts.output_path + "/outputs/animals_picker", model_name)

checkpoint_directory, image_directory = make_result_folders(output_directory)

iterations = trainer.picker_resume(checkpoint_directory,
                            hp=config,
                            multigpus=opts.multigpus)

while True:
    for it, (co_data, cl_data) in enumerate(zip(train_loader, test_loader)):
        with Timer("Elapsed time in update: %f"):
            loss = trainer.traffic_picker_update(co_data, cl_data, config)
            print('loss: %.4f' % (loss))

        if (iterations + 1) % config['log_iter'] == 0:
            print("Iteration: %08d/%08d" % (iterations + 1, max_iter))

        if (iterations + 1) % config['snapshot_save_iter'] == 0:
            print('change checkpoint dir', checkpoint_directory)
            trainer.save(checkpoint_directory, iterations, opts.multigpus)
            print('Saved model at iteration %d' % (iterations + 1))

        iterations += 1
        if iterations >= max_iter:
            print("Finish Training")
            sys.exit(0)
