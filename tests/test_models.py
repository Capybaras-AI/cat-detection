from functools import reduce
import torch
import unittest
from src.cnn_model import CNNModel

class TestsCNNModel(unittest.TestCase):
  def test_init_model(self):
    mock_input = torch.rand((3, 32, 32))
    kernel_size = 5
    output_size = 10
    stride = 1
    padding = 2
    model = CNNModel(mock_input.shape, output_size, kernel_size, 1, 2)
    # Actual number of parameters
    num_parameters = 0
    for param in model.parameters():
      num_parameters += reduce(lambda acum, x: x * acum, param.size(), 1)
    # Expected number of parameters
    exp_num_parameters = 0
    ## First hidden layer: 1 kernel and 1 bias per channel
    exp_num_parameters += (kernel_size ** 2 * 3 + 1) * 16
    hidden1_width = (mock_input.size(1) - kernel_size + 2 * padding) // stride + 1
    ## Second hidden layer: 1 kernel and 1 bias per channel
    exp_num_parameters += (kernel_size ** 2 * 16 + 1) * 8
    hidden2_width = (hidden1_width - kernel_size + 2 * padding) // stride + 1
    ## Output layer: 1 weight and 1 bias per output
    hidden2_size = (hidden2_width ** 2) * 8
    exp_num_parameters += (hidden2_size + 1) * output_size
    self.assertEqual(num_parameters, exp_num_parameters)
  
  def test_output_size(self):
    input_shape = (3, 32, 32)
    batch_size = (1,)
    mock_input = torch.rand(batch_size + input_shape)
    output_size = 10
    model = CNNModel(input_shape, output_size, 5, 1, 2)
    logits, probs, label = model.predict(mock_input)
    self.assertEqual(logits.size(), batch_size + (output_size,))
    self.assertEqual(probs.size(), batch_size + (output_size,))
    self.assertEqual(label.size(), batch_size)
