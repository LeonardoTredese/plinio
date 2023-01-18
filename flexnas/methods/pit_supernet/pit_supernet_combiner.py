from typing import List, cast, Iterator, Tuple
import torch
import torch.nn as nn
import torch.fx as fx
from torchinfo import summary
from flexnas.methods.pit import PITModule


class PITSuperNetCombiner(nn.Module):
    """This nn.Module is included in the PITSuperNetModule and contains most of the logic of
    the module and the NAS paramaters.

    :param input_layers: list of possible alternative layers to be selected
    :type input_layers: List[nn.Module]
    """
    def __init__(self, input_layers: List[nn.Module]):
        super(PITSuperNetCombiner, self).__init__()

        self.sn_input_layers = input_layers
        self.input_shape = None
        self.n_layers = len(input_layers)
        self.alpha = nn.Parameter(
            (1 / self.n_layers) * torch.ones(self.n_layers, dtype=torch.float), requires_grad=True)

        self._pit_layers = list()
        for _ in range(self.n_layers):
            self._pit_layers.append(list())

        self.layers_sizes = []
        self.layers_macs = []

    def forward(self, layers_outputs: List[torch.Tensor]):
        """Forward function for the PITSuperNetCombiner that returns a weighted
        sum of all the outputs of the different alternative layers.

        :param layers_outputs: outputs of all different modules
        :type layers_outputs: torch.Tensor
        :return: the output tensor (weighted sum of all layers output)
        :rtype: torch.Tensor
        """
        soft_alpha = nn.functional.softmax(self.alpha, dim=0)
        y = []
        for i, yi in enumerate(layers_outputs):
            y.append(soft_alpha[i] * yi)
        y = torch.stack(y, dim=0).sum(dim=0)
        return y

    def export(self, n: fx.Node, mod: fx.GraphModule):
        """Overwrites the module of a fx.Node corresponding to the PITSuperNetCombiner,
        with the layer selected by the NAS within a fx.GraphModule

        :param n: the node to be rewritten, corresponds to the sn_combiner node
        :type n: fx.Node
        :param mod: the parent module, where the new node has to be inserted
        :type mod: fx.GraphModule
        """
        index = torch.argmax(self.alpha).item()
        submodule = self.sn_input_layers[int(index)]
        target = str(n.target)
        mod.add_submodule(target, submodule)

    def compute_layers_sizes(self):
        """Computes the size of each possible layer of the PITSuperNetModule
        and stores the values in a list.
        It removes the size of the PIT modules contained in each layer because
        these sizes will be computed and re-added at training time.
        """
        for layer in self.sn_input_layers:
            layer_size = 0
            for param in layer.parameters():
                layer_size += torch.prod(torch.tensor(param.shape))
            self.layers_sizes.append(layer_size)

        for i, layer in enumerate(self.sn_input_layers):
            for module in layer.modules():
                if isinstance(module, PITModule):
                    cast(List[PITModule], self._pit_layers[i]).append(module)
                    # self._pit_layers[i].append(module)
                    size = 0
                    for param in module.parameters():
                        size += torch.prod(torch.tensor(param.shape))
                    self.layers_sizes[i] = self.layers_sizes[i] - size

    def compute_layers_macs(self):
        """Computes the MACs of each possible layer of the PITSuperNetModule
        and stores the values in a list.
        It removes the MACs of the PIT modules contained in each layer because
        these MACs will be computed and re-added at training time.
        """
        for layer in self.sn_input_layers:
            stats = summary(layer, self.input_shape, verbose=0, mode='eval')
            self.layers_macs.append(stats.total_mult_adds)

        for i, layer in enumerate(self.sn_input_layers):
            for module in layer.modules():
                if isinstance(module, PITModule):
                    # cast(List[PITModule], self._pit_layers[i]).append(module)
                    # already done in compute_layers_sizes
                    stats = summary(module, self.input_shape, verbose=0, mode='eval')
                    self.layers_macs[i] = self.layers_macs[i] - stats.total_mult_adds

    def get_size(self) -> torch.Tensor:
        """Method that returns the number of weights for the module
        computed as a weighted sum of the number of weights of each layer.

        :return: number of weights of the module (weighted sum)
        :rtype: torch.Tensor
        """
        soft_alpha = nn.functional.softmax(self.alpha, dim=0)

        size = torch.tensor(0, dtype=torch.float32)
        for i in range(self.n_layers):
            var_size = torch.tensor(0, dtype=torch.float32)
            for pl in cast(List[PITModule], self._pit_layers[i]):
                var_size += pl.get_size()
            size = size + (soft_alpha[i] * (self.layers_sizes[i] + var_size))
        return size

    def get_macs(self) -> torch.Tensor:
        """Method that computes the number of MAC operations for the module

        :return: the number of MACs
        :rtype: torch.Tensor
        """
        soft_alpha = nn.functional.softmax(self.alpha, dim=0)

        macs = torch.tensor(0, dtype=torch.float32)
        for i in range(self.n_layers):
            var_macs = torch.tensor(0, dtype=torch.float32)
            for pl in cast(List[PITModule], self._pit_layers[i]):
                var_macs += pl.get_macs()
            macs = macs + (soft_alpha[i] * (self.layers_macs[i] + var_macs))
        return macs

    def named_nas_parameters(
            self, prefix: str = '', recurse: bool = False) -> Iterator[Tuple[str, nn.Parameter]]:
        """Returns an iterator over the architectural parameters of this module, yielding
        both the name of the parameter as well as the parameter itself

        :param prefix: prefix to prepend to all parameter names, defaults to ''
        :type prefix: str, optional
        :param recurse: kept for uniformity with pytorch API, defaults to False
        :type recurse: bool, optional
        :yield: an iterator over the architectural parameters of all layers of the module
        :rtype: Iterator[Tuple[str, nn.Parameter]]
        """
        prfx = prefix
        prfx += "." if len(prefix) > 0 else ""
        prfx += "alpha"
        yield prfx, self.alpha

    def nas_parameters(self, recurse: bool = False) -> Iterator[nn.Parameter]:
        """Returns an iterator over the architectural parameters of this module

        :param recurse: kept for uniformity with pytorch API, defaults to False
        :type recurse: bool, optional
        :yield: an iterator over the architectural parameters of all layers of the module
        :rtype: Iterator[nn.Parameter]
        """
        for _, param in self.named_nas_parameters(recurse=recurse):
            yield param
