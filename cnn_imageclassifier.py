import cv2
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision
import torchvision.datasets as datasets
from torch.autograd import Variable
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import CNN


# transformation for image
transform_ori = transforms.Compose([
   transforms.RandomResizedCrop(64), #create 64x64 image
   transforms.RandomHorizontalFlip(), # flipping image horizontally
   transforms.ToTensor(), #convert image to Tensor
   transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]) #normalize image

# Load dataset
train_dataset = datasets.ImageFolder(root = 'dataset/training_set', transform = transform_ori)

test_dataset = datasets.ImageFolder(root = 'dataset/test_set', transform = transform_ori)

# make dataset iterable
batch_size = 65 # changed from 100
train_load = torch.utils.data.DataLoader(
   dataset = train_dataset,
   batch_size = batch_size,
   shuffle = True) # shuffle to create mixed batches of receipt and non-receipt images

test_load = torch.utils.data.DataLoader(
   dataset = test_dataset,
   batch_size = batch_size,
   shuffle = False)

print('{} images in training set'.format(len(train_dataset)))
print('{} images in training set'.format(len(test_dataset)))
print('{} batches in train loader'.format(len(train_load)))
print('{} batches in test loader'.format(len(test_load)))

model = CNN.CNN()

CUDA = torch.cuda.is_available()
if CUDA:
   print("CUDA available") # looks like it's not available
   model = model.cuda()
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr = 0.01)

# Training the CNN
#%%time
import time

num_epochs = 10

# define the lists to store the results of loss and accuracy
train_loss = []
test_loss = []
train_acc = []
test_acc = []

# Training
for epoch in range(num_epochs):
   # reset these below variables to 0 at beginning of every epoch
   start = time.time()
   correct = 0
   iterations = 0
   iter_loss = 0.0

   model.train() # put network into training mode

   for i, (inputs, labels) in enumerate(train_load):
      # convert torch tensor to Variable
      inputs = Variable(inputs)
      labels = Variable(labels)

      # if we have GPU, shift data to GPU
      CUDA = torch.cuda.is_available()
      if CUDA:
         inputs = inputs.cuda()
         labels = labels.cuda()

      optimizer.zero_grad() # clear off gradient in (w = w- gradient)
      outputs = model(inputs)
      loss = loss_fn(outputs, labels)
      iter_loss += loss.data # accumulate loss
      loss.backward() # backpropagation
      optimizer.step() # update the weights

      # record the correct predictions for training data
      _, predicted = torch.max(outputs, 1)
      correct += (predicted == labels).sum()
      iterations += 1

      # record the training loss
      train_loss.append(iter_loss/iterations)
      # record the training accuracy
      train_acc.append((100 * correct / len(train_dataset)))

      # testing
      loss = 0.0
      correct = 0
      iterations = 0

      model.eval() # put the network into evaluation mode

      for i, (inputs, labels) in enumerate(test_load):
         # convert torch tensor to Variable
         inputs = Variable(inputs)
         labels = Variable(labels)
         #print(labels)

         CUDA = torch.cuda.is_available()
         if CUDA:
            inputs = inputs.cuda()
            labels = labels.cuda()

         outputs = model(inputs)
         loss = loss_fn(outputs, labels) # calculate loss
         loss += loss.data
         # record the correct predictions for training data
         _, preicted = torch.max(outputs, 1)
         #print(len(predicted))
         #print(len(labels))
         correct += (predicted == labels).sum()
         
         iterations += 1

      # record testing loss
      test_loss.append(loss/iterations)
      # record testing accuracy
      test_acc.append((100 * correct / len(test_dataset)))
      stop = time.time()

      print('Epoch {}/{}, Training Loss: {:.3f}, Training Accuracy: {:.3f}, Testing Loss: {:.3f}, Testing Acc: {:.3f}, Time: {}s'
           .format(epoch+1, num_epochs, train_loss[-1], train_acc[-1], test_loss[-1], test_acc[-1], stop-start))

# Loss
f = plt.figure(figsize=(10, 10))
plt.plot(train_loss, label='Training Loss')
plt.plot(test_loss, label='Testing Loss')
plt.legend()
plt.savefig('loss.png')

# Accuracy
f = plt.figure(figsize=(10, 10))
plt.plot(train_acc, label='Training Accuracy')
plt.plot(test_acc, label='Testing Accuracy')
plt.legend()
plt.savefig('accuracy.png')

# save model
torch.save(model.state_dict(), 'classify_receipts.pth')
