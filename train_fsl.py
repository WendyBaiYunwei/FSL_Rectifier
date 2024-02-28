import numpy as np
import torch
from model.utils import (
    pprint, set_gpu,
    get_command_line_parser,
    postprocess_args,
)
from model.image_translator.utils import get_train_loaders, get_config

if __name__ == '__main__':
    parser = get_command_line_parser()
    args = postprocess_args(parser.parse_args())
    config = get_config('animals.yaml')
    config['max_iter'] == args.max_epoch * args.episodes_per_epoch
    config['way_size'] = args.way
    config['batch_size'] = args.query + args.shot
    pprint(vars(args))

    from model.trainer.fsl_trainer_animals import FSLTrainer
    # from model.trainer.fsl_trainer_traffic import FSLTrainer

    set_gpu(args.gpu)
    trainer = FSLTrainer(args, config)
    trainer.train()
    print(args.save_path)



