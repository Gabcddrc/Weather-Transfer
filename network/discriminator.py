import torch.nn
import torch.nn as nn
import torch
import torch.nn.functional as F
import torchvision.models as models
from torchvision import transforms as T
from seg_utils import deeplabv3plus_mobilenet, set_bn_momentum
from torch.nn.functional import one_hot

def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
        if hasattr(m, "bias") and m.bias is not None:
            torch.nn.init.constant_(m.bias.data, 0.0)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)


class Discriminator(torch.nn.Module):

    def __init__(self, CKPT_PATH, num_classes=19, output_stride=16):
        super(Discriminator, self).__init__()

        # pytorch model of 21 classes: not compatible with the 19 classes segmap!
        # self.deeplabv3_preprocessing = transforms.Normalize(mean=[0.485, 0.456, 0.406],
        #                                              std=[0.229, 0.224, 0.225])
        # self.semantic_seg_model = models.segmentation.deeplabv3_resnet50(pretrained = True)

        # deeplabv3plus model of 19 classes
        self.deeplabv3_preprocessing = T.Compose([
                        #T.ToTensor(),
                        T.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225]),
                    ])


        model = deeplabv3plus_mobilenet(num_classes, output_stride=output_stride,
                                        pretrained_backbone=True)

        set_bn_momentum(model.backbone, momentum=0.01)

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print("Device: %s" % device)

        checkpoint = torch.load(CKPT_PATH, map_location=torch.device('cpu'))
        model.load_state_dict(checkpoint["model_state"])
        #model = torch.nn.DataParallel(model)
        model.to(device)
        del checkpoint
        
        self.semantic_seg_model = model.eval()

        for param in self.semantic_seg_model.parameters():
            param.requires_grad = False

        resnet_18 = models.resnet18(pretrained=True)
        for param in resnet_18.parameters():
            param.requires_grad = False

        self.feature_extraction_network = resnet_18

        self.classification_layers = [
            torch.nn.utils.parametrizations.spectral_norm(torch.nn.Linear(1000, 512)),
            torch.nn.LeakyReLU(),
            torch.nn.BatchNorm1d(512),
            torch.nn.utils.parametrizations.spectral_norm(torch.nn.Linear(512, 256)),
            torch.nn.LeakyReLU(),
            torch.nn.utils.parametrizations.spectral_norm(torch.nn.Linear(256, 2)),
            torch.nn.Softmax(1)
        ]
        #self.classification_head = torch.nn.Sequential(*classification_layers)
        self.seq = torch.nn.ModuleList(self.classification_layers)

    def forward(self, X):
        """
        returns:

            B x 21 x H x W semantic map (
            https://pytorch.org/hub/pytorch_vision_deeplabv3_resnet101/)
            B x 2 - two vector if true or false regardin classification
        """
        # semantic_map = torch.zeros(1,1,1,1)
        # out = torch.zeros(1,2)
        with torch.no_grad():
            processed_X = self.deeplabv3_preprocessing(X)
            semantic_map = self.semantic_seg_model(processed_X)#['out']
            pred = semantic_map.max(1)[1]
            pred = one_hot(pred, num_classes=19).permute(0,3,1,2)

        features = self.feature_extraction_network(X)

        out = features
        for lay in self.seq:
            out = lay(out)

        return pred.float(), out


class Discriminator2(nn.Module):
    def __init__(self, CKPT_PATH, input_shape):
        super(Discriminator2, self).__init__()

        channels, height, width = input_shape

        # Calculate output shape of image discriminator (PatchGAN)
        self.output_shape = (1, height // 2 ** 4, width // 2 ** 4)

        self.deeplabv3_preprocessing = T.Compose([
                        #T.ToTensor(),
                        T.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225]),
                    ])


        model = deeplabv3plus_mobilenet(19, output_stride=16,
                                        pretrained_backbone=True)

        set_bn_momentum(model.backbone, momentum=0.01)

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print("Device: %s" % device)

        checkpoint = torch.load(CKPT_PATH, map_location=torch.device('cpu'))
        model.load_state_dict(checkpoint["model_state"])
        model = torch.nn.DataParallel(model)
        model.to(device)
        del checkpoint
        
        self.semantic_seg_model = model.eval()

        for param in self.semantic_seg_model.parameters():
            param.requires_grad = False


        def discriminator_block(in_filters, out_filters, normalize=False):
            """Returns downsampling layers of each discriminator block"""
            layers = [torch.nn.utils.parametrizations.spectral_norm(nn.Conv2d(in_filters, out_filters, 4, stride=2, padding=1))]
            if normalize:
                layers.append(nn.InstanceNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *discriminator_block(channels, 64, normalize=False),
            *discriminator_block(64, 128),
            *discriminator_block(128, 256),
            *discriminator_block(256, 512),
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(512, 1, 4, padding=1)
        )

    def forward(self, img):

        with torch.no_grad():
            processed_X = self.deeplabv3_preprocessing(img)
            semantic_map = self.semantic_seg_model(processed_X)#['out']
            pred = semantic_map.max(1)[1]
            pred = one_hot(pred, num_classes=19).permute(0,3,1,2)

        return pred.float(), self.model(img)
