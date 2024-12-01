import unittest
from src.dataset import CifarDataset

class TestsCifarDataset(unittest.TestCase):
  def test_init_train(self):
    dataset = CifarDataset()
    self.assertEqual(len(dataset), 50000, "Train size should be 50,000")
    self.assertEqual(dataset.images[0].shape, (3,32,32), "Train images should have shape 3x32x32")

  def test_init_test(self):
    dataset = CifarDataset(train=False)
    self.assertEqual(len(dataset), 10000, "Test size should be 10,000")
    self.assertEqual(dataset.images[0].shape, (3,32,32), "Test images should have shape 3x32x32")
    
if __name__ == '__main__':
  unittest.main()