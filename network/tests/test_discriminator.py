# some content with DeepLearning2021/test/Test_discriminator.ipynb
import torch
from torchinfo import summary
import unittest
from ..network.discriminator import Discriminator

# +
CKPT_PATH = '/psi/home/li_s1/data/Season/pretrained/best_deeplabv3plus_mobilenet_cityscapes_os16.pth'
class DiscriminatorTest(unittest.TestCase):
    def test_real_fake_1(self):
        network = Discriminator(CKPT_PATH)
        print(network)
        pytorch_train_total_params = sum(p.numel() for p in network.parameters() if p.requires_grad)
        pytorch_total_params = sum(p.numel() for p in network.parameters())
        print("trainable params in model: ", pytorch_train_total_params)
        print("total params in model: ", pytorch_total_params)
        summary(network, (3, 320, 320))

    def test_real_fake_2(self):
        network = Discriminator(CKPT_PATH)
        map, out = network(torch.zeros(2, 3, 320, 320))
        print(map.shape, out.shape)
        #self.assertEqual((2, 19, 320, 320), map.shape)
        #self.assertEqual((2, 2), out.shape)

if __name__ == '__main__':
    unittest.main()
