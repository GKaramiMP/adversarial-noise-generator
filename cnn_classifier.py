import logging
import torchvision
import torchvision.models as models
from torchvision.models import VGG16_Weights
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Callable
import matplotlib.pyplot as plt

#######################################

class CNNClassifier:
    def __init__(self,
                 learning_rate: float = 0.001,
                 max_epochs: int = 200,
                 batch_size: int = 128,
                 num_classes: int = 10,
                 log_dir: Optional[str] = None,
                 dropout_prob: float = 0.5,
                 dataset: Optional[object] = None,
                 restore_model_path: Optional[str] = None):
        """"
         CNNClassifier: A general-purpose wrapper for CNN-based models like VGG16.
        Args: 
            learning_rate (float): Learning rate for training.
            max_epochs (int): Number of epochs for training.
            batch_size (int): Size of each batch during training.
            num_classes (int): Number of output classes.
            log_dir (str, optional): Directory to save logs.
            dropout_prob (float): Dropout probability in the model.
            dataset (object, optional): Data loader or dataset object.
            restore_model_path (str, optional): Path to restore a pre-trained model.
        """

        self.learning_rate = learning_rate  
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.log_dir = log_dir
        self.dropout_prob = dropout_prob
        self.restore_model_path = restore_model_path
        self.dataset = dataset
    

    def vgg_model(self,
                  in_channels: int,
                  num_classes: int,
                  weights: VGG16_Weights = VGG16_Weights.IMAGENET1K_V1,
                  checkpoint_path: Optional[str]= None):
        """
        Modifies the VGG16 model to adapt to custom input channels and output classes.
        Args:
            in_channels (int): Number of input channels (e.g., 1 for grayscale).
            num_classes (int): Number of output classes.
            weights (VGG16_Weights): Pretrained weights to use.
            checkpoint_path (str, optional): Path to load pre-trained model weights.
        Returns:
            nn.Module: Modified VGG16 model.
        """
        
        logging.info("Initializing VGG16 model...")
            
        model_vgg = models.vgg16(weights)

        # Modify the first convolutional layer for input channels
        original_conv1 = model_vgg.features[0]
        model_vgg.features[0] = nn.Conv2d(in_channels=in_channels,
                                          out_channels=original_conv1.out_channels,
                                         kernel_size=original_conv1.kernel_size,
                                         stride=original_conv1.stride,
                                         padding=original_conv1.padding,
                                         bias=original_conv1.bias is not None, )

        # Copy weights from the original layer and adjust
        with torch.no_grad():
            model_vgg.features[0].weight[:, 0] = original_conv1.weight.mean(dim=1)  # Average weights across RGB channels

        # Modify the classifier for custom output classes
        model_vgg.classifier[6] = nn.Linear(4096, num_classes)


        if checkpoint_path:
            checkpoint = torch.load(checkpoint_path)
            self.model_vgg.load_state_dict(checkpoint)

        print("VGG16 model restored and modified for", num_classes, "classes.")
        return model_vgg
    
    
    @staticmethod
    def update_noise(noise, noise_limit, step_size, grad, fast_sign, targeted_mod):
        
        if fast_sign:
            if targeted_mod:
                noise -= step_size * np.sign(grad)
            else:
                noise += step_size * np.sign(grad)
        else:
            if targeted_mod:
                noise -= step_size * grad / max(np.abs(grad.max()), np.abs(grad.min()))
            else:
                noise += step_size * grad / max(np.abs(grad.max()), np.abs(grad.min()))
        noise = np.clip(noise, -noise_limit, noise_limit)
        return noise

    def generate_adversarial_noise(self,
                                   model: nn.Module, 
                                   criterion: nn.Module, 
                                   images: torch.Tensor, 
                                   labels: torch.Tensor, 
                                   cls_target: Optional[int] = None, 
                                   epsilon: float = 0.01, 
                                   targeted: bool = False
                                  ):
        """
        Generates adversarial examples for a batch of images.
        Args:
            model (nn.Module): The model to attack.
            criterion (nn.Module): Loss function.
            images (torch.Tensor): Input tensor of images.
            labels (torch.Tensor): Ground truth labels.
            cls_target (int, optional): Target class for a targeted attack.
            epsilon (float): Perturbation size.
            targeted (bool): Whether the attack is targeted or not.
                targeted=False: The goal is to make the model misclassify the image into any wrong class
        Returns:
            torch.tensor: Adversarial examples.
        """
        
        model.eval() # Set model to evaluation mode
        outputs = model(images)
        # print("Outputs grad_fn:", outputs.grad_fn)
    
        if targeted:
            loss = criterion(outputs, torch.full_like(labels, cls_target))
        else:
            loss = criterion(outputs, labels)
     
        model.zero_grad()
        loss.backward()
    
        gradient_sign = images.grad.sign()
        if targeted:
            noise = -epsilon * gradient_sign
        else:
            noise = epsilon * gradient_sign

        adversarial_images = torch.clamp(images + noise, 0, 1)
        logging.info("Adversarial noise generated.")
        
        return adversarial_images
    