import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class MaskGenerator(nn.Module):

    def __init__(self, num_primary_color):
        super(MaskGenerator, self).__init__()
        in_dim = 3 + num_primary_color * 3 
        out_dim = num_primary_color 
        self.conv1 = nn.Conv2d(in_dim, in_dim * 2, kernel_size=3, stride=2, padding=1, bias=False)
        self.conv2 = nn.Conv2d(in_dim * 2, in_dim * 4, kernel_size=3, stride=2, padding=1, bias=False)
        self.conv3 = nn.Conv2d(in_dim * 4, in_dim * 8, kernel_size=3, stride=2, padding=1, bias=False)
        self.deconv1 = nn.ConvTranspose2d(in_dim * 8, in_dim * 4, kernel_size=3, stride=2, padding=1, bias=False, output_padding=1)
        self.deconv2 = nn.ConvTranspose2d(in_dim * 8, in_dim * 2, kernel_size=3, stride=2, padding=1, bias=False, output_padding=1)
        self.deconv3 = nn.ConvTranspose2d(in_dim * 4, in_dim * 2, kernel_size=3, stride=2, padding=1, bias=False, output_padding=1)
        self.conv4 = nn.Conv2d(in_dim * 2 + 3, in_dim, kernel_size=3, stride=1, padding=1)
        self.conv5 = nn.Conv2d(in_dim, out_dim, kernel_size=3, stride=1, padding=1)

        self.bn1 = nn.BatchNorm2d(in_dim * 2)
        self.bn2 = nn.BatchNorm2d(in_dim * 4)
        self.bn3 = nn.BatchNorm2d(in_dim * 8)
        self.bnde1 = nn.BatchNorm2d(in_dim * 4)
        self.bnde2 = nn.BatchNorm2d(in_dim * 2)
        self.bnde3 = nn.BatchNorm2d(in_dim * 2)
        self.bn4 = nn.BatchNorm2d(in_dim)

    def forward(self, target_img, primary_color_pack):
        x = torch.cat((target_img, primary_color_pack), dim=1)

        h1 = self.bn1(F.relu(self.conv1(x)))
        h2 = self.bn2(F.relu(self.conv2(h1))) 
        h3 = self.bn3(F.relu(self.conv3(h2))) 
        h4 = self.bnde1(F.relu(self.deconv1(h3))) 
        h4 = torch.cat((h4, h2), 1) 
        h5 = self.bnde2(F.relu(self.deconv2(h4))) 
        h5 = torch.cat((h5, h1), 1) 
        h6 = self.bnde3(F.relu(self.deconv3(h5))) 
        h6 = torch.cat((h6, target_img), 1) 
        h7 = self.bn4(F.relu(self.conv4(h6)))

        return torch.sigmoid(self.conv5(h7)) 

class ResiduePredictor(nn.Module):
    def __init__(self, num_primary_color):
        super(ResiduePredictor, self).__init__()

        in_dim = 3 + num_primary_color * 4
        out_dim = num_primary_color * 3

        self.conv1 = nn.Conv2d(in_dim, in_dim * 2, kernel_size=3, stride=2, padding=1, bias=False)
        self.conv2 = nn.Conv2d(in_dim * 2, in_dim * 4, kernel_size=3, stride=2, padding=1, bias=False)
        self.conv3 = nn.Conv2d(in_dim * 4, in_dim * 8, kernel_size=3, stride=2, padding=1, bias=False)
        self.deconv1 = nn.ConvTranspose2d(in_dim * 8, in_dim * 4, kernel_size=3, stride=2, padding=1, bias=False, output_padding=1)
        self.deconv2 = nn.ConvTranspose2d(in_dim * 8, in_dim * 2, kernel_size=3, stride=2, padding=1, bias=False, output_padding=1)
        self.deconv3 = nn.ConvTranspose2d(in_dim * 4, in_dim * 2, kernel_size=3, stride=2, padding=1, bias=False, output_padding=1)
        self.conv4 = nn.Conv2d(in_dim * 2 + 3, in_dim, kernel_size=3, stride=1, padding=1)
        self.conv5 = nn.Conv2d(in_dim, out_dim, kernel_size=3, stride=1, padding=1)

        self.bn1 = nn.BatchNorm2d(in_dim * 2)
        self.bn2 = nn.BatchNorm2d(in_dim * 4)
        self.bn3 = nn.BatchNorm2d(in_dim * 8)
        self.bnde1 = nn.BatchNorm2d(in_dim * 4)
        self.bnde2 = nn.BatchNorm2d(in_dim * 2)
        self.bnde3 = nn.BatchNorm2d(in_dim * 2)
        self.bn4 = nn.BatchNorm2d(in_dim)

    def forward(self, target_img, mono_color_layers_pack):
        x = torch.cat((target_img, mono_color_layers_pack), dim=1)

        h1 = self.bn1(F.relu(self.conv1(x))) 
        h2 = self.bn2(F.relu(self.conv2(h1))) 
        h3 = self.bn3(F.relu(self.conv3(h2))) 
        h4 = self.bnde1(F.relu(self.deconv1(h3))) 
        h4 = torch.cat((h4, h2), 1) 
        h5 = self.bnde2(F.relu(self.deconv2(h4))) 
        h5 = torch.cat((h5, h1), 1) 
        h6 = self.bnde3(F.relu(self.deconv3(h5)))
        h6 = torch.cat((h6, target_img), 1) 
        h7 = self.bn4(F.relu(self.conv4(h6)))
        residue_pack = torch.tanh(self.conv5(h7))
        residue_pack = residue_pack - residue_pack.mean(dim=2, keepdim=True).mean(dim=3, keepdim=True)
        
        return residue_pack 
