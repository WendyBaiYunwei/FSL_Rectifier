
import torch
import os
import sys
import argparse
import shutil

from model.image_translator.utils import get_config, get_train_loaders, make_result_folders, \
    write_loss, write_html, write_1images, Timer
from model.image_translator.trainer import Translator_Trainer

import torch.backends.cudnn as cudnn

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
parser.add_argument("--resume",
                    action="store_true")
opts = parser.parse_args()

# Load experiment setting
config = get_config(opts.config)
max_iter = config['max_iter']
# Override the batch size if specified.
if opts.batch_size != 0:
    config['batch_size'] = opts.batch_size

trainer = Translator_Trainer(config)
trainer.cuda()
if opts.multigpus:
    ngpus = torch.cuda.device_count()
    config['gpus'] = ngpus
    print("Number of GPUs: %d" % ngpus)
    trainer.model = torch.nn.DataParallel(
        trainer.model, device_ids=range(ngpus))
else:
    config['gpus'] = 1

loaders = get_train_loaders(config)
train_content_loader = loaders[0]
train_class_loader = loaders[1]
test_content_loader = loaders[2]
test_class_loader = loaders[3]

model_name = os.path.splitext(os.path.basename(opts.config))[0]
output_directory = os.path.join(opts.output_path + "/outputs", model_name)
checkpoint_directory, image_directory = make_result_folders(output_directory)
shutil.copy(opts.config, os.path.join(output_directory, 'config.yaml'))

iterations = trainer.resume(checkpoint_directory,
                            hp=config,
                            multigpus=opts.multigpus) if opts.resume else 0

while True:
    for it, (co_data, cl_data) in enumerate(
            zip(train_content_loader, train_class_loader)):
        with Timer("Elapsed time in update: %f"):
            d_acc = trainer.dis_update(co_data, cl_data, config)
            g_acc = trainer.gen_update(co_data, cl_data, config,
                                       opts.multigpus)
            torch.cuda.synchronize()
            print('D acc: %.4f\t G acc: %.4f' % (d_acc, g_acc))

        if (iterations + 1) % config['log_iter'] == 0:
            print("Iteration: %08d/%08d" % (iterations + 1, max_iter))
            # write_loss(iterations, trainer, train_writer)

        if ((iterations + 1) % config['image_save_iter'] == 0 or (
                iterations + 1) % config['image_display_iter'] == 0):
            if (iterations + 1) % config['image_save_iter'] == 0:
                key_str = '%08d' % (iterations + 1)
                write_html(output_directory + "/index.html", iterations + 1,
                           config['image_save_iter'], 'images')
            else:
                key_str = 'current'
            with torch.no_grad():
                for t, (val_co_data, val_cl_data) in enumerate(
                        zip(train_content_loader, train_class_loader)):
                    if t >= opts.test_batch_size:
                        break
                    val_image_outputs = trainer.test(val_co_data, val_cl_data,
                                                     opts.multigpus)
                    write_1images(val_image_outputs, image_directory,
                                  'train_%s_%02d' % (key_str, t))

        if (iterations + 1) % config['snapshot_save_iter'] == 0:
            trainer.save(checkpoint_directory, iterations, opts.multigpus)
            print('Saved model at iteration %d' % (iterations + 1))

        iterations += 1
        if iterations >= max_iter:
            print("Finish Training")
            sys.exit(0)