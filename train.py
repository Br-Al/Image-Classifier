import argparse
from utility import train, data_loader, save_checkpoint
import torch
from model import Model

argp = argparse.ArgumentParser()

argp.add_argument('-d', '--dataset', required = True, help = 'Dataset Directory path')

argp.add_argument('-o', '--save_dir', help = 'directory to save checkpoints', default = './' )
argp.add_argument('-a', '--arch', help = 'Model architecture', default = 'vgg19',  choices = ['inception_v3', 'resnext101_32x8d', 'resnet101', 'resnet152', 'mobilenet_v3_large', 'vgg19', 'squeezenet', 'alexnet', 'resnet101'])
argp.add_argument('-l', '--lr', help = 'Learning Rate', default = 0.001, type = float)
argp.add_argument('--hidden_units', help = 'Hidden units', type = int, default = 512)
argp.add_argument('-e', '--epochs', help = 'Epcochs', type = int, default = 30)
argp.add_argument('--device', help = 'gpu or cpu', choices = ['cpu', 'cuda'], default = 'cuda')

arg = vars(argp.parse_args())
epochs = arg['epochs']
learning_rate = arg['lr']
save_dir = arg['save_dir']
criterion = torch.nn.NLLLoss()
model = Model(arg['arch'])
model = model.set_classifier(arg['hidden_units'])
optimizer = torch.optim.Adam(model.classifier.parameters(), lr=learning_rate)
if torch.cuda.is_available():
    device = torch.device(arg['device'])
else:
    device = torch.device('cpu')

dataloaders = data_loader(arg['dataset'])
model = train(model, epochs, criterion, optimizer, dataloaders['train'], dataloaders['valid'], device)
save_checkpoint(save_dir, model, learning_rate, epochs, arg['arch'])
