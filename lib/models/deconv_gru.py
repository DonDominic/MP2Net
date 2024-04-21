import torch
import torch.nn as nn
import torch.nn.functional as F
from lib.models.DCNv2.dcn_v2 import DCN

class DeConvGRUCell(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size, bias=True, activation=F.tanh):
        """
        Initialize ConvGRU cell.
        Parameters
        ----------
        input_dim: int
            Number of channels of input tensor.
        hidden_dim: int
            Number of channels of hidden state.
        kernel_size: (int, int)
            Size of the convolutional kernel.
        bias: bool
            Whether or not to add the bias.
        """
        super(DeConvGRUCell, self).__init__()
        self.input_dim          = input_dim
        self.hidden_dim         = hidden_dim

        self.kernel_size = kernel_size
        self.padding     = kernel_size[0] // 2, kernel_size[1] // 2
        self.bias        = bias
        self.activation  = activation

        # self.down = nn.Conv2d(self.input_dim, self.hidden_dim, kernel_size=1, stride=1, padding=0, bias=False)
        # self.bn0 = nn.BatchNorm2d(self.hidden_dim)

        self.conv_z = nn.Conv2d(in_channels=self.input_dim + self.hidden_dim,
                              out_channels=self.hidden_dim * 2,
                              kernel_size=self.kernel_size,
                              padding=self.padding,
                              bias=self.bias)
        
        self.conv_r = nn.Conv2d(in_channels=self.input_dim + self.hidden_dim,
                              out_channels=self.hidden_dim,
                              kernel_size=self.kernel_size,
                              padding=self.padding,
                              bias=self.bias)

        self.conv_h1 = nn.Conv2d(in_channels=self.input_dim,
                              out_channels=self.hidden_dim,
                              kernel_size=self.kernel_size,
                              padding=self.padding,
                              bias=self.bias)

        self.conv_h2 = nn.Conv2d(in_channels=self.hidden_dim,
                              out_channels=self.hidden_dim,
                              kernel_size=self.kernel_size,
                              padding=self.padding,
                              bias=self.bias)

        self.reset_parameters()

    def forward(self, input, h_prev):
        _, _, H, W = input.shape
        combined = torch.cat((input, h_prev), dim=1)  # concatenate along channel axis

        r = F.sigmoid(self.conv_r(combined))
        z = F.sigmoid(self.conv_z(combined))

        z1, z2 = torch.split(z, self.hidden_dim, dim=1)

        h_ = self.activation(self.conv_h1(input) + r * self.conv_h2(h_prev))

        h_cur = z1 * h_ + z2 * h_prev

        return h_cur

    def init_hidden(self, batch_size, height, width, cuda=True):
        state = torch.zeros(batch_size, self.hidden_dim, height, width)
        if cuda:
            state = state.cuda()
        return state

    def reset_parameters(self):
        #self.conv.reset_parameters()
        nn.init.xavier_uniform_(self.conv_z.weight, gain=nn.init.calculate_gain('sigmoid'))
        self.conv_z.bias.data.zero_()
        nn.init.xavier_uniform_(self.conv_r.weight, gain=nn.init.calculate_gain('sigmoid'))
        self.conv_r.bias.data.zero_()
        nn.init.xavier_uniform_(self.conv_h1.weight, gain=nn.init.calculate_gain('tanh'))
        self.conv_h1.bias.data.zero_()
        nn.init.xavier_uniform_(self.conv_h2.weight, gain=nn.init.calculate_gain('tanh'))
        self.conv_h2.bias.data.zero_()


class DeConvGRU(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size, num_layers, batch_first=True, bias=True, activation=F.tanh):
        super(DeConvGRU, self).__init__()

        self._check_kernel_size_consistency(kernel_size)

        # Make sure that both `kernel_size` and `hidden_dim` are lists having len == num_layers
        kernel_size = self._extend_for_multilayer(kernel_size, num_layers)
        hidden_dim  = self._extend_for_multilayer(hidden_dim, num_layers)
        activation  = self._extend_for_multilayer(activation, num_layers)

        if not len(kernel_size) == len(hidden_dim) == len(activation) == num_layers:
            raise ValueError('Inconsistent list length.')

        self.input_dim  = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.bias = bias

        cell_list = []
        for i in range(0, self.num_layers):
            cur_input_dim = self.input_dim if i == 0 else self.hidden_dim[i-1]

            cell_list.append(DeConvGRUCell(input_dim=cur_input_dim,
                                        hidden_dim=self.hidden_dim[i],
                                        kernel_size=self.kernel_size[i],
                                        bias=self.bias,
                                        activation=activation[i]
                                        ))

        self.cell_list = nn.ModuleList(cell_list)

        self.reset_parameters()

    def forward(self, input, hidden_state=None):
        """
        Parameters
        ----------
        input_tensor:
            5-D Tensor either of shape (t, b, c, h, w) or (b, t, c, h, w)
        hidden_state:
        Returns
        -------
        last_state_list, layer_output
        """
        cur_layer_input = torch.unbind(input, dim=int(self.batch_first))

        if hidden_state is None:
            hidden_state = self.get_init_states(cur_layer_input[0].size(0), cur_layer_input[0].size(2), cur_layer_input[0].size(3))

        seq_len = len(cur_layer_input)

        layer_output_list = []
        last_state_list   = []

        for layer_idx in range(self.num_layers):
            h = hidden_state[layer_idx]
            output_inner = []
            for t in range(seq_len):
                h = self.cell_list[layer_idx](input=cur_layer_input[t], h_prev=h)
                output_inner.append(h)

            cur_layer_input = output_inner
            last_state_list.append(h)

        layer_output = torch.stack(output_inner, dim=int(self.batch_first))

        return layer_output, h

    def reset_parameters(self):
        for c in self.cell_list:
            c.reset_parameters()
    
    def get_init_states(self, batch_size, height, width, cuda=True):
        init_states = []
        for i in range(self.num_layers):
            init_states.append(self.cell_list[i].init_hidden(batch_size, height, width, cuda))
        return init_states

    @staticmethod
    def _check_kernel_size_consistency(kernel_size):
        if not (isinstance(kernel_size, tuple) or (isinstance(kernel_size, list)
            and all([isinstance(elem, tuple) for elem in kernel_size]))):
            raise ValueError('`kernel_size` must be tuple or list of tuples')

    @staticmethod
    def _extend_for_multilayer(param, num_layers):
        if not isinstance(param, list):
            param = [param] * num_layers
        return param