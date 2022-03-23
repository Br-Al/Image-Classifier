from utility import process_image, predict
import argparse
import json
import torch
from utility import load_checkpoint

argp = argparse.ArgumentParser()

argp.add_argument('-i', '--img_path', required = True, help = 'path to the input image')

argp.add_argument('-c', '--checkpoint', help = 'Path to the checkpoint: .pth file', required = True)
argp.add_argument('-k', '--top_k', help = 'top K most likely classes', default = 5, type = int)
argp.add_argument('--category_names', help = 'mapping of categories to real names', default = './cat_to_name.json')
argp.add_argument('--device', help = 'gpu or cpu', choices = ['cpu', 'cuda'], default = 'cpu')

arg = vars(argp.parse_args())

with open(arg['category_names'], 'r') as f:
    cat_to_name = json.load(f)
    
    
if torch.cuda.is_available():
    device = arg['device']
else:
    device = 'cpu'

try:
    image = process_image(arg['img_path'])
except FileNotFoundError:
    print('Sorry we can not load your image!')
    
result = predict(image, arg['checkpoint'], cat_to_name, arg['top_k'], device)

print(result)
