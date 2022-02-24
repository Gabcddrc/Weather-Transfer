from ..generator_attention_module import *
import torch
import unittest
from torchsummary import summary
from ..generator_translation_module import *
from ..generator import Generator
from tqdm import tqdm

class TestAttentionLayer(unittest.TestCase):
    def test_visualize_attention_unit(self):
        summary(Self_Attention_2d(8), (8, 50, 50), device="cpu")

    def test_dimension_1(self):
        attention_layer = Self_Attention_2d(8)
        fake_input = torch.zeros(1, 8, 50, 50)
        output = attention_layer(fake_input)
        self.assertEqual(fake_input.shape, output[0].shape)
        pass

    def test_dimension_2(self):
        attention_layer = Self_Attention_2d(8)
        fake_input = torch.zeros(8, 8, 50, 50)
        output = attention_layer(fake_input)
        self.assertEqual(fake_input.shape, output[0].shape)
        pass

    def test_dimension_3(self):
        attention_layer = Self_Attention_2d(16)
        fake_input = torch.zeros(8, 16, 50, 50)
        output = attention_layer(fake_input)
        self.assertEqual(fake_input.shape, output[0].shape)
        pass


class TestDownscaleUnit(unittest.TestCase):
    def test_visualize_downscale_unit(self):
        summary(DownscaleUnit(4, 16, 4), (4, 64, 64), device="cpu")

    def test_downscale_unit(self):
        downscale_unit = DownscaleUnit(4, 16, 4)
        input_tensor = torch.zeros(1, 4, 64, 64)
        output = downscale_unit(input_tensor)
        self.assertEqual((1, 16, 32, 32), output.shape)
        pass

    def test_downscale_unit_2(self):
        downscale_unit = DownscaleUnit(4, 16, 4)
        input_tensor = torch.zeros(4, 4, 64, 64)
        output = downscale_unit(input_tensor)
        self.assertEqual((4, 16, 32, 32), output.shape)
        pass

    def test_downscale_unit_3(self):
        downscale_unit = DownscaleUnit(8, 16, 8)
        input_tensor = torch.zeros(2, 8, 128, 720)
        output = downscale_unit(input_tensor)
        self.assertEqual((2, 16, 64, 360), output.shape)
        pass

    def test_downscale_unit_4(self):
        downscale_unit = DownscaleUnit(8, 16, 3)
        input_tensor = torch.zeros(2, 8, 128, 720)
        output = downscale_unit(input_tensor)
        self.assertEqual((2, 16, 64, 360), output.shape)
        pass

    def test_downscale_unit_5(self):
        downscale_unit = DownscaleUnit(8, 16, 3)
        input_tensor = torch.zeros(2, 8, 320, 180)
        output = downscale_unit(input_tensor)
        self.assertEqual((2, 16, 160, 90), output.shape)
        pass

    def test_downscale_unit_6(self):
        downscale_unit = DownscaleUnit(8, 16, 3)
        input_tensor = torch.zeros(2, 8, 180, 320)
        output = downscale_unit(input_tensor)
        self.assertEqual((2, 16, 90, 160), output.shape)
        pass

    def test_downscale_unit_7(self):
        downscale_unit = DownscaleUnit(8, 16, 3)
        input_tensor = torch.zeros(2, 8, 512, 224)
        output = downscale_unit(input_tensor)
        self.assertEqual((2, 16, 256, 112), output.shape)
        pass

    def test_downscale_unit_8(self):
        downscale_unit = DownscaleUnit(8, 16, 3)
        max_res = 512
        for w in range(100, max_res, 8):
            for h in range(w, max_res, 8):
                input_tensor = torch.zeros(2, 8, h, w)
                output = downscale_unit(input_tensor)
                with self.subTest(line=(h,w)):
                    self.assertEqual((2, 16, h//2, w//2), output.shape)


class TestUpscaleUnit(unittest.TestCase):
    def test_visualize_upscale_unit(self):
        summary(UpscaleUnit(16, 4, 3), (16, 64, 64), device='cpu')
        # crashes

    def test_upscale_unit(self):
        upscale_unit = UpscaleUnit(16, 4, 3)
        input_tensor = torch.zeros(1, 16, 64, 64)
        output = upscale_unit(input_tensor)
        self.assertEqual((1, 4, 128, 128), output.shape)
        pass  # works

    def test_upscale_unit_2(self):
        upscale_unit = UpscaleUnit(16, 4, 3, padding=1, output_padding=1)
        input_tensor = torch.zeros(4, 16, 64, 64)
        output = upscale_unit(input_tensor)
        self.assertEqual((4, 4, 128, 128), output.shape)
        pass

    def test_upscale_unit_3(self):
        upscale_unit = UpscaleUnit(16, 8, 7, padding=int(8 / 2))
        input_tensor = torch.zeros(2, 16, 128, 720)
        output = upscale_unit(input_tensor)
        self.assertEqual((2, 8, 256, 1440), output.shape)
        pass

    def test_upscale_unit_4(self):
        upscale_unit = UpscaleUnit(16, 16, 3)
        input_tensor = torch.zeros(2, 16, 32, 32)
        output = upscale_unit(input_tensor)
        self.assertEqual((2, 16, 64, 64), output.shape)
        pass

    def test_upscale_unit_5(self):
        upscale_unit = UpscaleUnit(8, 4, 3)
        max_res = 512
        for w in (range(20, max_res, 8)):
            for h in (range(w, max_res, 8)):
                input_tensor = torch.zeros(2, 8, h, w)
                output = upscale_unit(input_tensor)
                with self.subTest(line=(h, w)):
                    self.assertEqual((2, 4, h*2, w*2), output.shape)


class TestResNet(unittest.TestCase):
    def test_resnet(self):
        network = ResNet(in_channels=4, kernel_size=3, blocks_sizes=[64, 128, 256])
        summary(network, (4, 224, 224), device="cpu")


class TestInitialTranslationNetwork(unittest.TestCase):
    def test_resnet(self):
        network = InitialTranslationNetwork(in_channels=4, blocks_sizes=[64, 128, 256],
                                            kernel_size_upsampling=3, kernel_size=3)
        summary(network, (4, 224, 224), device="cpu")


class TestAttentionNetwork(unittest.TestCase):
    def test_visualize(self):
        network = AttentionNetwork(in_dim=(1, 4, 64, 64),
                                   downscale_units=1,
                                   attention_units_sequence=1,
                                   attention_units_width=1,
                                   output_channels=1)
        # print("the network is", network)
        # summary(network, (4,64,64), device="cpu")

        self.assertTrue(True)

    def test_visualize_2(self):
        network = AttentionNetwork(in_dim=(1, 4, 640, 360),
                                   downscale_units=4,
                                   attention_units_sequence=5,
                                   attention_units_width=1,
                                   output_channels=1)
        #        print("the network is", network)
        summary(network, (4, 64, 64), device="cpu")

        self.assertTrue(True)

    def test_attention_1(self):
        network = AttentionNetwork(in_dim=(1, 4, 64, 64),
                                   downscale_units=1,
                                   attention_units_sequence=1,
                                   attention_units_width=1,
                                   output_channels=1)

        input = torch.rand(1, 4, 64, 64)
        output = network(input)
        self.assertEqual((1, 1, 64, 64), output.shape)

    def test_attention_2(self):
        network = AttentionNetwork(in_dim=(1, 4, 64, 64),
                                   downscale_units=2,
                                   attention_units_sequence=1,
                                   attention_units_width=1,
                                   output_channels=1)

        input = torch.rand(1, 4, 64, 64)
        output = network(input)
        self.assertEqual((1, 1, 64, 64), output.shape)

    def test_attention_2_4(self):
        network = AttentionNetwork(in_dim=(1, 4, 64, 64),
                                   downscale_units=4,
                                   attention_units_sequence=1,
                                   attention_units_width=1,
                                   output_channels=1)

        input = torch.rand(1, 4, 64, 64)
        output = network(input)
        self.assertEqual((1, 1, 64, 64), output.shape)

    def test_attention_3(self):
        network = AttentionNetwork(in_dim=(1, 4, 64, 64),
                                   downscale_units=1,
                                   attention_units_sequence=1,
                                   attention_units_width=1,
                                   output_channels=2)

        input = torch.rand(1, 4, 64, 64)
        output = network(input)
        self.assertEqual((1, 2, 64, 64), output.shape)

    def test_attention_4(self):
        network = AttentionNetwork(in_dim=(1, 4, 64, 64),
                                   downscale_units=1,
                                   attention_units_sequence=1,
                                   attention_units_width=2,
                                   output_channels=1)

        input = torch.rand(1, 4, 64, 64)
        output = network(input)
        self.assertEqual((1, 1, 64, 64), output.shape)

    def test_attention_5(self):
        network = AttentionNetwork(in_dim=(1, 4, 64, 64),
                                   downscale_units=1,
                                   attention_units_sequence=2,
                                   attention_units_width=1,
                                   output_channels=1)

        input = torch.rand(1, 4, 64, 64)
        output = network(input)
        self.assertEqual((1, 1, 64, 64), output.shape)

    def test_attention_6(self):
        network = AttentionNetwork(in_dim=(1, 4, 128, 720),
                                   downscale_units=4,
                                   attention_units_sequence=3,
                                   attention_units_width=2,
                                   output_channels=4)

        input = torch.rand(1, 4, 128, 720)
        output = network(input)
        self.assertEqual((1, 4, 128, 720), output.shape)

    def test_attention_7(self):
        network = AttentionNetwork(in_dim=(1, 24, 128, 720),
                                   downscale_units=4,
                                   attention_units_sequence=3,
                                   attention_units_width=2,
                                   output_channels=4)

        input = torch.rand(1, 24, 128, 720)
        output = network(input)
        self.assertEqual((1, 4, 128, 720), output.shape)


class TestGenerator(unittest.TestCase):

    def test_generator_1(self):
        network = Generator(in_channels_transl=4,
                            block_sizes_transl=[64, 128, 256],
                            kernel_size_transl=3,
                            kernel_size_upsampl_transl=3,
                            input_shape_attention=(1, 4, 320, 180),
                            attention_module_output_channels=1,
                            kernel_size_blending=(3, 3),
                            kernel_size_final_filter=(3, 3))
        summary(network, (4, 180, 320), device="cpu")

    def test_generator_2(self):
        network = Generator(in_channels_transl=4,
                            block_sizes_transl=[64, 128, 256],
                            kernel_size_transl=3,
                            kernel_size_upsampl_transl=3,
                            input_shape_attention=(1, 4, 320, 180),
                            attention_module_output_channels=1,
                            kernel_size_blending=(3, 3),
                            kernel_size_final_filter=(3, 3))
        out = network(torch.zeros(1, 4, 180, 320))
        self.assertEqual((1, 3, 180, 320), out)

    # def test_generator_2(self):
    #     network = Generator((1, 4, 320, 180), 1)
    #     out = network(torch.zeros(1, 4, 320, 180))
    #     self.assertEqual((1, 3, 320, 180), out)
    #
    # def test_generator_3(self):
    #     network = Generator((4, 4, 320, 180), 1)
    #     out = network(torch.zeros(4, 4, 320, 180))
    #     self.assertEqual((4, 3, 320, 180), out)
    #
    # def test_generator_4(self):
    #     network = Generator((4, 4, 540, 540), 1)
    #     out = network(torch.zeros(4, 4, 540, 540))
    #     self.assertEqual((4, 3, 540, 540), out)
    #
    def test_generator_4(self):
        input_shape = (1, 22, 256, 256)
        network = Generator(in_channels_transl=input_shape[1],
                              block_sizes_transl=[64, 128, 256],
                              kernel_size_transl=3,
                              kernel_size_upsampl_transl=3,
                              input_shape_attention=input_shape,
                              attention_module_output_channels=1,
                              kernel_size_blending=(3, 3),
                              kernel_size_final_filter=(3, 3))
        out = network(torch.zeros(input_shape))
        self.assertEqual((4, 3, 256, 256), out.shape)

    def test_generator_5(self):
        input_shape = (1, 22, 256, 448)
        network = Generator(in_channels_transl=input_shape[1],
                            block_sizes_transl=[64, 128, 256],
                            kernel_size_transl=3,
                            kernel_size_upsampl_transl=3,
                            input_shape_attention=input_shape,
                            attention_module_output_channels=1,
                            kernel_size_blending=(3, 3),
                            kernel_size_final_filter=(3, 3))
        out = network(torch.zeros(input_shape))
        self.assertEqual((1, 3, 256, 448), out.shape)

    def test_generator_6(self):
        input_shape = (1, 22, 224, 400)
        network = Generator(in_channels_transl=input_shape[1],
                            block_sizes_transl=[64, 128, 256],
                            kernel_size_transl=3,
                            kernel_size_upsampl_transl=3,
                            input_shape_attention=input_shape,
                            attention_module_output_channels=1,
                            kernel_size_blending=(3, 3),
                            kernel_size_final_filter=(3, 3))
        out = network(torch.zeros(input_shape))
        self.assertEqual((1, 3, 224, 400), out.shape)

    def test_generator_6(self):
        input_shape = (1, 22, 400, 224)
        network = Generator(in_channels_transl=input_shape[1],
                            block_sizes_transl=[64, 128, 256],
                            kernel_size_transl=3,
                            kernel_size_upsampl_transl=3,
                            input_shape_attention=input_shape,
                            attention_module_output_channels=1,
                            kernel_size_blending=(3, 3),
                            kernel_size_final_filter=(3, 3))
        out = network(torch.zeros(input_shape))
        self.assertEqual((1, 3, 400, 224), out.shape)

    def test_model_size(self):
        input_shape = (1, 22, 288, 512)

        generator = Generator(in_channels_transl=input_shape[1],
                              block_sizes_transl=[64, 128, 256],
                              kernel_size_transl=3,
                              kernel_size_upsampl_transl=3,
                              input_shape_attention=input_shape,
                              attention_module_output_channels=1,
                              kernel_size_blending=(3, 3),
                              kernel_size_final_filter=(3, 3))
        pytorch_train_total_params = sum(p.numel() for p in generator.parameters() if p.requires_grad)
        pytorch_total_params = sum(p.numel() for p in generator.parameters())
        print("trainable params in model: ", pytorch_train_total_params)
        print("total params in model: ", pytorch_total_params)

        input = torch.zeros(input_shape)
        output = generator(input)
        self.assertEqual((1, 3, 288, 512), output.shape)
        summary(generator, (22, 288, 512), device="cpu")

if __name__ == "__main__":
    unittest.main()
