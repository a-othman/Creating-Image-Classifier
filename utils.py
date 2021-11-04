import argparse
import json
from PIL import Image
from collections import OrderedDict
from workspace_utils import active_session
from tqdm import tqdm_notebook as tqdm
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt

import torch
from torch import nn
from torch import optim
from torchvision import datasets, transforms, models
import torch.nn.functional as F


#parsing arguments from the CLI
def get_input():
    
    parser = argparse.ArgumentParser() #creating a parser
    parser.add_argument('--data_dir', type=str)
    parser.add_argument('--save_dir', type=str, default= '/home/workspace/ImageClassifier/trained_model/checkpoint.pth')
    
    parser.add_argument('--arch', default= 'vgg16')
    parser.add_argument('--learning_rate', type=float, default= .001)
    parser.add_argument('--hidden_units', type=int)
    parser.add_argument('--epochs', type=int, default= 3)
#     parser.add_argument('--gpu', type=str, default= 'gpu')
    parser.add_argument('--gpu', type=bool, default= True)
    return parser.parse_args()
#loading data from directory
def load_data(data_dir):
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'
    #Define your transforms for the training, validation, and testing sets
    train_transformations = transforms.Compose([transforms.RandomRotation(30),
                                                transforms.RandomResizedCrop(224),
                                                transforms.RandomHorizontalFlip(),
                                                transforms.ToTensor(),
                                                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    transformations = transforms.Compose([
                              transforms.Resize(255),
                              transforms.CenterCrop(224),
                              transforms.ToTensor(),
                              transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                             ])
#     Load the datasets with ImageFolder
#     image_datasets= datasets.ImageFolder(data_dir, transform= transformations) 
    train = datasets.ImageFolder(train_dir, transform= train_transformations) 
    valid= datasets.ImageFolder(valid_dir, transform= transformations)
    test= datasets.ImageFolder(test_dir, transform= transformations)
    
    #Using the image datasets and the trainforms, define the dataloaders
    trainloader = torch.utils.data.DataLoader(train, batch_size= 64, shuffle=True)
    validloader= torch.utils.data.DataLoader(valid, batch_size= 64)
    testloader= torch.utils.data.DataLoader(test, batch_size= 64)
    return trainloader, validloader, testloader



##I am still figuring out how to write the function of the model
def model_building(arch, gpu, hidden_units):
    
#     model = models.arch(pretrained=True)
    model = getattr(models, arch)
    model = model(pretrained=True)
    for parm in model.parameters():
        parm.requires_grad= False
    if arch== 'vgg16':
        input_size= 25088
        if hidden_units == None:
            hidden_units= 4096
    elif arch== 'vgg13':
        input_size= 25088
        if hidden_units == None:
            hidden_units= 512
    else:
        print("Please propoer model either vgg16 or vgg13")
 
    classifier= nn.Sequential(OrderedDict([
                        ('fc1', nn.Linear(input_size, hidden_units)),
                        ('relu', nn.ReLU()),
                        ('dropout', nn.Dropout(0.3)),
                        ('output', nn.Linear(hidden_units, 102)),
                         ('log_softmax', nn.LogSoftmax(dim=1))])) 
    model.classifier= classifier
    if gpu== True:
        device= 'cuda'
        model.cuda()
    else:
        device= 'cpu'
        model.cpu()
    
    return model

# def train(data_dir, arch, hidden_units, epochs, learning_rate, device, gpu):
def train(data_dir, model, device, epochs, learning_rate,criterion, optimizer):   
    trainloader, validloader, testloader = load_data(data_dir)    
    training_errors= []
    validation_error= []
    steps= 0
    print_every= 1
    
    for epoch in range(epochs):
        
        train_loss= 0
        model.train()
        
        for images, labels in tqdm(trainloader, desc='train'):
            
            steps+=1
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            output= model.forward(images)
            loss= criterion(output, labels)
            loss.backward()
            optimizer.step()
            
            train_loss+=loss.item()
            
        if steps%print_every==0:
            
            validation_loss, accuracy = validation(model, validloader, device, criterion)
            training_errors.append(train_loss/len(trainloader))
            validation_error.append(validation_loss)
            print(f"Epochs {epoch+1}/{epochs}.."
                 f"Train loss {train_loss/len(trainloader):.3f}.."
                 f"validation loss {validation_loss/len(validloader):.3f}.."
                 f"Accuracy {accuracy/len(validloader):.3f}")
            
    print("Finished Training")        
        
        
def validation(model, validloader, device,criterion):
    validation_loss= 0
    accuracy= 0
    model.eval()
    with torch.no_grad():
#         for images, labels in validloader:
        for images, labels in tqdm(validloader, desc='Validation'):
            images, labels = images.to(device), labels.to(device)
            output= model.forward(images)
            loss= criterion(output, labels)
            validation_loss+= loss
            _, predicted = torch.max(output.data, 1)
            equals= (predicted== labels.view(*predicted.shape))
            accuracy += torch.mean(equals.type(torch.cuda.FloatTensor)).item()
        validation_error= validation_loss/len(validloader)
        
    return validation_error, accuracy


def save_model(path_save, arch, model, data_dir, optimizer):
    train_transformations = transforms.Compose([transforms.RandomRotation(30),
                                                transforms.RandomResizedCrop(224),
                                                transforms.RandomHorizontalFlip(),
                                                transforms.ToTensor(),
                                                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    train_dir = data_dir + '/train'
    train_data = datasets.ImageFolder(train_dir, transform = train_transformations)
    
    model.class_to_idx = train_data.class_to_idx

    input_size =    25088 if arch== 'vgg16' else 1024 
    checkpoint = {'transfer_model': arch,
                'input_size': input_size,
                'output_size': 102,
                'features': model.features,
                'classifier': model.classifier,
                'optimizer': optimizer.state_dict(),
                'state_dict': model.state_dict(),
                'idx_to_class': {v: k for k, v in model.class_to_idx.items()}
                        }
    torch.save(checkpoint, path_save)
    

    
def get_input_4_prediction():
    parser = argparse.ArgumentParser() #creating a parser

    parser.add_argument('--img_path', type=str)
    model_checkpoint_dir = '/home/workspace/ImageClassifier/trained_model/checkpoint.pth'
    parser.add_argument('--checkpoint', type=str, default= model_checkpoint_dir)
    parser.add_argument('--top_k', type= int, default= 1)
    parser.add_argument('--category_names_path', type= str, default= None)
    parser.add_argument('--gpu', type=bool, default= True)
    return parser.parse_args()
    
#loading the labels mapping from category label to category name
def cat_to_name(category_names_path):
    with open(category_names_path, 'r') as f:
        cat_to_name = json.load(f)
    return cat_to_name

#Write a function that loads a checkpoint and rebuilds the model
def load_model(save_dir):
    checkpoint = torch.load(save_dir)
    arch = checkpoint['transfer_model']
    if arch== 'vgg16':
        model= models.vgg16(pretrained=True)
        input_size= 25088
        hidden_units= 4096
    elif arch== 'vgg13':
        model= models.vgg16(pretrained=True)
        input_size= 25088
        hidden_units= 512
        
    for parm in model.parameters():
        torch.requires_grad= False
    model.classifier= nn.Sequential(OrderedDict([
                        ('fc1', nn.Linear(input_size, hidden_units)),
                        ('relu', nn.ReLU()),
                        ('dropout', nn.Dropout(0.3)),
                        ('output', nn.Linear(hidden_units, 102)),
                         ('log_softmax', nn.LogSoftmax(dim=1))]))
    
    model.load_state_dict(checkpoint['state_dict'])
    return model, checkpoint



def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    image= Image.open(image)
    w, h = image.size
    image.thumbnail((256, 256*(h/w)))

    w, h= image.size #256 x 256*(old_h/old_w)
    left= (w - 224)/2
    top = (h - 224)/2
    right= (w+ 224)/2
    bottom= (h + 224)/2
    image= image.crop((left, top, right, bottom))
    
    image = np.array(image).astype('float64')
    image= image/ [255, 255, 255]
    mean= [0.485, 0.456, 0.406]
    std= [0.229, 0.224, 0.225]
    image= (image - mean)/std 
    image= image.transpose((2, 0, 1))
    
    return image

def predict(image_path, model_checkpoint_dir, topk=5):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    image= process_image(image_path)
    image= torch.from_numpy(image).unsqueeze_(0).float()
    
    model, model_details= load_model(model_checkpoint_dir)
    output= model(image)
    prob, classes= torch.exp(output).topk(topk)

    return prob.detach().numpy(), classes[0].add(1).tolist()


