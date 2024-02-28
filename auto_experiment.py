import os
import torch
import argparse
import subprocess
assert torch.cuda.is_available(), "\033[31m You need GPU to Train! \033[0m"
print("CPU Count is :", os.cpu_count())

paths = [
    'animals_conv4_checkpoint.pth',
    'traffic_conv4_checkpoint.pth'
]

commands = [
   f"python test_fsl.py --model_class ProtoNet--backbone_class ConvNet --dataset Animals --num_eval_episodes 1000 --model_path {paths[0]} --spt_expansion 1 --qry_expansion 0 --add_transform original",
   f"python test_fsl.py --model_class ProtoNet--backbone_class ConvNet --dataset Animals --num_eval_episodes 1000 --model_path {paths[0]} --spt_expansion 1 --qry_expansion 0 --add_transform crop+rotate",
   f"python test_fsl.py --model_class ProtoNet--backbone_class ConvNet --dataset Traffic --num_eval_episodes 1000 --model_path {paths[1]} --spt_expansion 1 --qry_expansion 0 --add_transform original",
   "python get_embedding.py", 
   "python TSNE.py",   
]

for command in commands:
   print(f"Command: {command}")

   process = subprocess.Popen(command, shell=True)
   outcode = process.wait()
   if (outcode):
      break
