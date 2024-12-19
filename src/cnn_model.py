import torch
import torch.nn as nn
import torch.functional as F
import numpy as np

class CNNModel(nn.Module):
  """
  Model with two convolutional layers and a final linear (MPL) that
  takes as input 3x32x32 images and outputs class probabilities
  """
  def __init__(self, input_shape, num_classes, kernel_size, stride, padding, device = "cpu"):
    super(CNNModel, self).__init__()
    # Input dimensions
    input_channels, input_width, input_height = input_shape
    self.conv1 = nn.Conv2d(in_channels=input_channels, out_channels=16, kernel_size=kernel_size, stride=stride, padding=padding)
    # First hidden layer
    hidden1_width = (input_width-kernel_size+2*padding) // stride + 1
    hidden1_height = (input_height-kernel_size+2*padding) // stride + 1
    # Second hidden layer
    self.conv2 = nn.Conv2d(in_channels=16, out_channels=8, kernel_size=kernel_size, stride=stride, padding=padding)
    hidden2_width = (hidden1_width-kernel_size+2*padding) // stride + 1
    hidden2_height = (hidden1_height-kernel_size+2*padding) // stride + 1
    self.flatten = nn.Flatten()
    # Output layer
    hidden2_flat_size = hidden2_width * hidden2_height * 8
    self.linear = nn.Linear(in_features=hidden2_flat_size, out_features=num_classes)
    self.softmax = nn.Softmax(dim=0)
    # Loss
    self.cross_entropy = nn.CrossEntropyLoss()
    if device == "cuda":
      self.cuda()
    return

  def forward(self, X, y):
    """
    Computes the predicted probabilities for each class and the loss for the given input `X` and target value `y`

    :param torch.Tensor X: Batch of input images with values [0, 1] and shape (batch size, channels, width, height)
    :param torch.Tensor y: Target label with shape (batch size)
    :return torch.Tensor y_prob: Predicted probabilities with shape (batch size, classes)
    :return torch.Tensor loss: Loss with shape (1)
    """
    logits, probs, label = self.predict(X)[0]
    loss = self.cross_entropy(logits, y) # (b, o) -> (1)
    return loss, probs, label
  
  def predict(self, X):
    """
    Returns the label of the class with the highest probability given the input features `X`

    :param torch.Tensor X: Batch of input images with values [0, 1] and shape (batch size, channels, width, height)
    :result torch.Tensor y_hat: Class label with shape (batch size)
    """
    hidden1 = self.conv1(X) # (b, c, w, h) -> (b, c', w', h')
    hidden2 = self.conv2(hidden1) # (b, c', w', h') -> (b, c'', w'', h'')
    hidden2_flat = self.flatten(hidden2) # (b, c'', w'', h'') -> (b, c'' * w'' * h'')
    logits = self.linear(hidden2_flat) # (b, c'' * w'' * h'') -> (b, o)
    probs = self.softmax(logits) # (b, o) -> (b, o)
    y_hat = probs.argmax(dim=1) # (b, o) -> (b)
    return logits, probs, y_hat