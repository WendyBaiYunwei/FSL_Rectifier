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
    if args.dataset == 'Animals':
        config = get_config('animals.yaml')
    else:
        config = get_config('traffic.yaml')
    config['way_size'] = args.way
    config['batch_size'] = args.eval_query + args.eval_shot
    config['eval_shot'] = args.eval_shot
    config['eval_query'] = args.eval_query
    
    pprint(vars(args))

    from model.trainer.fsl_trainer_animals import FSLTrainer
    # from model.trainer.fsl_trainer_buffer import FSLTrainer

    set_gpu(args.gpu)
    trainer = FSLTrainer(args, config)
    trainer.evaluate_test()
    print(args.save_path)



