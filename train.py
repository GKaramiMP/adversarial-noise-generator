import os
import numpy as np
import time
from datetime import datetime

import torch
import torch.nn as nn
import torchvision
from torchvision import transforms
from torchvision.transforms import Compose
from torchvision.models import VGG16_Weights

import mnist_dataset
import cnn_classifier
from torch.utils.tensorboard import SummaryWriter

import matplotlib.pyplot as plt
plt.show(block=True)

#####################################################################
# Constants
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TRAIN_SUBSET_SIZE = 480
VALID_SUBSET_SIZE = 160
BATCH_SIZE = 32
MAX_EPOCHS = 10
CLS_TARGET = 5
NUM_CLASSES = 10
IN_CHANNELS = 1
LEARNING_RATE = 1e-5
WEIGTH_DECAY=1e-3
DROP_OUT_PROB = 0.5

LOG_DIR = '/home/ec2-user/SageMaker/MNIST/results/'
os.makedirs(LOG_DIR, exist_ok=True)

#####################################################################
def imshow(image_tensor, title=None):
    """
    Plots the first 16 images in a batch of image tensors.
    Args:
        image_tensor: Tensor of shape [B, C, H, W].
        title: Optional title for the plot (str).
    """
    
    if not isinstance(image_tensor, (torch.Tensor, np.ndarray)):
        raise TypeError("image_tensor must be a torch.Tensor or numpy.ndarray")

           
    images = image_tensor.detach().cpu().numpy() if isinstance(image_tensor, torch.Tensor) else image_tensor
    batch_size = images.shape[0]
    num_images = min(batch_size, 16) 

    fig, axes = plt.subplots(1, num_images, figsize=(15, 5))

    for i in range(num_images):
        image = images[i, 0]
        axes[i].imshow(image, cmap='gray')
        axes[i].axis('off')  
        if title:
            plt.suptitle(title)
    plt.tight_layout()
    plt.show()

####################################################################
if __name__ =="__main__":
    
    print('device:', device)
    writer = SummaryWriter(log_dir=LOG_DIR)

    ####################################################################
    ''' Setup of data pipeline '''
    
    train_transform = Compose([transforms.Resize((224, 224)),
                               transforms.ToTensor(),
                               transforms.Normalize((0.5,), (0.5,))])

    valid_transform = Compose([transforms.Resize((224, 224)),
                               transforms.ToTensor(),
                               transforms.Normalize((0.5,), (0.5,))])

    mnist_ds = mnist_dataset.MNISTDataset(batch_size=BATCH_SIZE, 
                                             train_transform=train_transform,
                                             valid_transform=valid_transform,
                                             train_subset_size=TRAIN_SUBSET_SIZE,
                                             valid_subset_size=VALID_SUBSET_SIZE)
    
    data_loaders = mnist_ds.get_data_loader(batch_size=BATCH_SIZE)
    train_loader = data_loaders['train']
    valid_loader = data_loaders['valid']

    ####################################################################
    ''' Initialize Model, Loss, Optimizer '''
    
    cnn = cnn_classifier.CNNClassifier(learning_rate=LEARNING_RATE,
                                           max_epochs=MAX_EPOCHS,
                                           batch_size=BATCH_SIZE,
                                           num_classes=NUM_CLASSES,
                                           log_dir=LOG_DIR + '/logs',  
                                           dropout_prob=0.5,   
                                           restore_model_path=None,  
                                           dataset=mnist_ds)
    
    weights = VGG16_Weights.DEFAULT  # Loads pretrained weights
    model = cnn.vgg_model(in_channels=IN_CHANNELS, num_classes=NUM_CLASSES, weights=weights)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGTH_DECAY)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)


    ####################################################################
    ''' start training '''
    
    val_interval = 1
    best_metric = -1
    best_metric_epoch = -1
    epoch_loss_values = []
    metric_values = []
    metric_values_adv = []
    best_accuracy = 0

    print('num_epochs:', MAX_EPOCHS)
    for epoch in range(MAX_EPOCHS):
        print(f"epoch {epoch + 1}/{MAX_EPOCHS}")
        
        model.train()
        
        epoch_loss = 0
        accuracies = []
        step = 0
        for batch_data in train_loader:
            step += 1
            images, labels = batch_data[0].to(device), batch_data[1].to(device)
            images.requires_grad = True
            labels = labels.long()

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            loss.backward()
            optimizer.step()

            preds = torch.argmax(outputs, dim=1)
            correct = torch.sum(preds == labels)
            accuracy = correct.double() / labels.size(0)

            accuracies.append(accuracy.item())
            epoch_loss += loss.item()
            
            print(f"{step}/{len(mnist_ds.train_dataset) // train_loader.batch_size}, " f"train_loss: {loss.item():.4f}, "f"train_acc:{accuracy.item():.2f}")
            epoch_len = len(mnist_ds.train_dataset) // train_loader.batch_size
            writer.add_scalar("train_loss", loss.item(), epoch_len * epoch + step)
            writer.add_scalar("train_acc", accuracy.item(), epoch_len * epoch + step)
            
        epoch_loss /= step
        epoch_loss_values.append(epoch_loss)
        avg_acc = np.mean(accuracies)
        lr_scheduler.step()

        if (epoch + 1) % val_interval == 0:
            model.eval()
            print('#'*50)
            print('Validation Phase')
            val_acc, val_acc_adv = 0.0, 0.0
            with torch.set_grad_enabled(True):
            # with torch.no_grad():
                for batch_data in valid_loader:
                    images, labels = batch_data[0].to(device), batch_data[1].to(device)
                    images.requires_grad = True
                    labels = labels.long()
                    
                    # Normal performance
                    outputs = model(images)
                    
                    preds = outputs.argmax(dim=1)
                    val_acc += (preds == labels).sum().item()

                    # Adversarial performance
                    adv_images = cnn.generate_adversarial_noise(model, criterion, images, labels, cls_target=CLS_TARGET, epsilon=0.01, targeted=False)
                    adv_outputs = model(adv_images)
                    adv_preds = adv_outputs.argmax(dim=1)
                    val_acc_adv += (adv_preds == labels).sum().item()

            val_acc /= len(valid_loader.dataset)
            val_acc_adv /= len(valid_loader.dataset)
            
            writer.add_scalar("Validation Accuracy", val_acc, epoch)
            writer.add_scalar("Validation Accuracy (Adv)", val_acc_adv, epoch)
            print(f"Validation Accuracy: {val_acc:.2f}")
            print(f"Adversarial Accuracy: {val_acc_adv:.2f}")
            

            # print(adv_images.shape)
            # adv_images = adv_images.cpu()
            # imshow(adv_images[:16], title="Adversarial Images")
            # print("Predicted Adv labels:", adv_preds[:16].cpu().numpy())

            # Save best model
            if val_acc_adv > best_accuracy:
                best_accuracy = val_acc_adv
                torch.save(model.state_dict(), os.path.join(LOG_DIR, "best_model.pth"))
                
            print('#'*50)
            print('EPOCH {} SUMMARY'.format(epoch+1))
            print('Training Phase.')
            print('  Average Train Loss:          {:.3f}'.format(epoch_loss))
            print('  Average Train Accuracy:      {:.2f}'.format(avg_acc))
            print('Validation Phase.')
            print('  Average Validation Accuracy:     {:.2f}'.format(val_acc))
            print('  Average Adversarial Accuracy:    {:.2f}'.format(val_acc_adv))
            print('#'*50)
                
    writer.close()
    print(f"Training complete. Best Adversarial Accuracy: {best_accuracy:.2f}")

