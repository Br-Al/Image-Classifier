import os
from torchvision import transforms, datasets
import torch
import torchvision
from PIL import Image
def data_loader(dataset_dir):
    train_dir = os.path.join(dataset_dir, 'train')
    valid_dir = os.path.join(dataset_dir, 'valid')
    test_dir = os.path.join(dataset_dir, 'test')
    
    data_transforms ={ 
        'train' :transforms.Compose([
            transforms.RandomRotation(degrees= (0, 90)),
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])]),
        'testAndValid' : transforms.Compose([
            transforms.Resize(size = (224,224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])]),}
    image_datasets = {
        'train' : datasets.ImageFolder(train_dir,transform = data_transforms['train']),
        'test' : datasets.ImageFolder(test_dir, transform = data_transforms['testAndValid']),
        'valid' : datasets.ImageFolder(valid_dir, transform = data_transforms['testAndValid'])}

    dataloaders = { 
        'train' : torch.utils.data.DataLoader(
            image_datasets['train'], 
            batch_size = 32,         
            shuffle = True),
        'test' : torch.utils.data.DataLoader(
            image_datasets['test'], 
            batch_size = 32 , 
            shuffle = True),
        'valid' : torch.utils.data.DataLoader(
            image_datasets['valid'], 
            batch_size = 32, 
            shuffle = True)}
    return dataloaders


def validate(model, criterion, data_loader, device = 'cpu'):
    model.eval() 
    accuracy = 0
    test_loss = 0
    total = 0
    for images, labels in iter(data_loader):
        model.to(device)
        
        output = model.forward(images.to(device))
        test_loss += criterion(output, labels.to(device)).item()
        ps = torch.exp(output)
        top_p, top_class = ps.topk(1, dim=1)
        equals = top_class == labels.to(device).view(*top_class.shape)
        accuracy += torch.mean(equals.type(torch.FloatTensor))
        total += labels.size(0)

    return test_loss/len(data_loader), accuracy/total

def train(model, 
          epochs, 
          criterion, 
          optimizer, 
          training_loader, 
          validation_loader, 
          device = 'cuda',):
    
    model.train() 
    
    print_every = 100
    steps = 0

    for epoch in range(epochs):
        running_loss = 0
        for images, labels in iter(training_loader):
            
            steps += 1
            
            model.to(device)
            optimizer.zero_grad() 
            
            output = model.forward(images.to(device)) 
            loss = criterion(output, labels.to(device)) 
            loss.backward() 
            optimizer.step()
            running_loss += loss.item()

            if steps % print_every == 0:
                validation_loss, accuracy = validate(model, criterion, validation_loader, device)

                print("Epoch: {}/{} ".format(epoch+1, epochs),
                        "Training Loss: {:.3f} ".format(running_loss/print_every),
                        "Validation Loss: {:.3f} ".format(validation_loss),
                        "Validation Accuracy: {:.3f}".format(accuracy))
                running_loss = 0
    
    return model
def save_checkpoint(path, model, lr, epochs, arch):
#     model.class_to_idx = image_datasets['train'].class_to_idx
    
    # Create model data dictionary
    if model.fc:
        checkpoint = {
                'output_size': 102,
                'arch': arch,
                'learning_rate': lr,
                'classifier' : model.fc,
                'epochs': epochs,
                'state_dict': model.state_dict(),
                }

    elif model.classifier:
        checkpoint = {
                'output_size': 102,
                'arch': arch,
                'learning_rate': lr,
                'classifier' : model.classifier,
                'epochs': epochs,
                'state_dict': model.state_dict(),
                }

    # Save to a file
    torch.save(checkpoint, os.path.join(path,'checkpoint.pth'))
    
def load_checkpoint(path, map_location):

    checkpoint = torch.load(path, map_location=map_location)
    model = getattr(torchvision.models, checkpoint['arch'])(pretrained=True)
    model.learning_rate = checkpoint['learning_rate']
    if model.fc:
        model.fc = checkpoint['classifier']
    elif model.classifier:
        model.classifier = checkpoint['classifier']
    model.epochs = checkpoint['epochs']
    model.load_state_dict(checkpoint['state_dict'])
        
    return model

def process_image(image_path):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    image = Image.open(image_path)
    transform = transforms.Compose([
        transforms.Resize(size=256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
    ])
    return transform(image) 

def predict(image, checkpoint, cat_to_name, topk=5, device = 'cpu'):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    model = load_checkpoint(checkpoint, device)
    model.to('cpu')
    
    output = model(image.unsqueeze(0))
    ps = torch.exp(output)
    top_p, top_class = ps.topk(topk, dim=1)
    top_class = [cat_to_name[str(cls_id)] for cls_id in top_class.numpy().tolist()[0]]
    probs = top_p.detach().cpu().numpy().tolist()[0]
    return top_class, probs