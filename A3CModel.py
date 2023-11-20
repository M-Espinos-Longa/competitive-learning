import torch
import torch.nn as nn

class A3CModel(nn.Module):
    """
    Advantage Actor Critic model. Original paper can be found on
    https://arxiv.org/abs/1602.01783v2
    """
    def model_init(self, num_obs_c, num_obs_h, num_obs_w, num_actions, num_hidden_units,
        kernels, strides, fmaps, device):
        """
        Model initialisation. Uses agents_info neural network data
        Inputs:
            + num_obs_c (int) -> observation channels
            + num_obs_h (int) -> height of image observation
            + num_obs_w (int) -> width of image observation
            + num_actions (int) -> number of available actions per agent
            + num_hidden_units (list) -> hidden units per layer
            + kernels (list) -> squared kernels on each conv layer
            + strides (list) -> strides on each conv layer
            + fmaps (list) -> feature maps in convolutional layers
            + device (string) -> computational resources allocation
        Outputs:
        """
        # define kernel sizes and strides of each convolutional layer
        k1 = kernels[0]
        k2 = kernels[1]
        k3 = kernels[2]
        s1 = strides[0]
        s2 = strides[1]
        s3 = strides[2]

        # compute input size of last linear fully connected layer (formula can be
        # found in https://pytorch.org/docs/1.9.1/generated/torch.nn.Conv2d.html)
        h1 = ((num_obs_h - (k1 - 1) - 1) // s1) + 1
        h2 = ((h1 - (k2 - 1) - 1) // s2) + 1
        h3 = ((h2 - (k3 - 1) - 1) // s3) + 1
        w1 = ((num_obs_w - (k1 - 1) - 1) // s1) + 1
        w2 = ((w1 - (k2 - 1) - 1) // s2) + 1
        w3 = ((w2 - (k3 - 1) - 1) // s3) + 1
        num_input_linear_layer = h3 * w3 * fmaps[2]

        # conv layers
        self.conv1 = nn.Conv2d(num_obs_c, fmaps[0], kernel_size=k1, stride=s1).to(device)
        self.conv2 = nn.Conv2d(fmaps[0], fmaps[1], kernel_size=k2, stride=s2).to(device)
        self.conv3 = nn.Conv2d(fmaps[1], fmaps[2], kernel_size=k3, stride=s3).to(device)

        # 2d normalisation layers
        #self.norm1 = nn.BatchNorm2d(fmaps[0]).to(device)
        #self.norm2 = nn.BatchNorm2d(fmaps[1]).to(device)
        #self.norm3 = nn.BatchNorm2d(fmaps[2]).to(device)

        # linear layers (2 streams, one for state value output and another for
        # policy terms)
        self.linear_v = nn.Linear(num_input_linear_layer, num_hidden_units[0]).to(device)
        self.value = nn.Linear(num_hidden_units[0], 1).to(device)
        self.linear_p = nn.Linear(num_input_linear_layer, num_hidden_units[0]).to(device)
        self.policy = nn.Linear(num_hidden_units[0], num_actions).to(device)

        # non-linear activation
        self.relu = nn.ReLU().to(device)

        # flatten layer
        self.flat = nn.Flatten().to(device)

    def forward_pass(self, input):
        """
        Inputs:
            + input (tensor) -> batch of osbervations
        Outputs:
            + val (tensor) -> state values
            + pol (tesnor) -> policy
        """
        # convolutions with normalisation
        outconv1 = self.relu(self.conv1(input))
        outconv2 = self.relu(self.conv2(outconv1))
        outconv3 = self.relu(self.conv3(outconv2))

        # flatten convolutional output
        flat = self.flat(outconv3)

        # value MLP
        outlin_v = self.relu(self.linear_v(flat))
        val = self.value(outlin_v)

        # policy MLP
        outlin_p = self.relu(self.linear_p(flat))
        pol = self.policy(outlin_p)

        return val, pol
