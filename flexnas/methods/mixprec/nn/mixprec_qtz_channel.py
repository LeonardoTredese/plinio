# *----------------------------------------------------------------------------*
# * Copyright (C) 2022 Politecnico di Torino, Italy                            *
# * SPDX-License-Identifier: Apache-2.0                                        *
# *                                                                            *
# * Licensed under the Apache License, Version 2.0 (the "License");            *
# * you may not use this file except in compliance with the License.           *
# * You may obtain a copy of the License at                                    *
# *                                                                            *
# * http://www.apache.org/licenses/LICENSE-2.0                                 *
# *                                                                            *
# * Unless required by applicable law or agreed to in writing, software        *
# * distributed under the License is distributed on an "AS IS" BASIS,          *
# * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.   *
# * See the License for the specific language governing permissions and        *
# * limitations under the License.                                             *
# *                                                                            *
# * Author:  Matteo Risso <matteo.risso@polito.it>                             *
# *----------------------------------------------------------------------------*

from typing import Dict, Tuple, Type, cast
import torch
import torch.nn as nn
from quant import Quantizer


class MixPrec_Qtz_Channel(nn.Module):
    """A nn.Module implementing a generic mixed-precision quantization searchable
    operation of each channel of tensor provided as input.
    This module includes trainable NAS parameters.

    :param precisions: different bitwitdth alternatives among which perform search
    :type precisions: Tuple[int, ...]
    :param cout: number of output channels
    :type cout: int
    :param quantizer: input quantizer
    :type quantizer: Quantizer
    :param quantizer_kwargs: quantizer kwargs, if no kwargs are passed default is used
    :type quantizer_kwargs: Dict, optional
    """
    def __init__(self,
                 precisions: Tuple[int, ...],
                 cout: int,
                 quantizer: Type[Quantizer],
                 quantizer_kwargs: Dict = {}):
        super(MixPrec_Qtz_Channel, self).__init__()
        self.precisions = precisions
        self.cout = cout
        self.quantizer = quantizer
        self.quantizer_kwargs = quantizer_kwargs
        # NAS parameters
        self.alpha_prec = nn.Parameter(torch.Tensor(len(precisions), cout))
        self.alpha_prec.data.fill_(1.)  # Initially each precision is equiprobable
        # Mixed-Precision Quantizers
        self.mix_qtz = nn.ModuleList()
        for p in precisions:
            qtz = quantizer(p, **quantizer_kwargs)  # type: ignore
            qtz = cast(nn.Module, qtz)
            self.mix_qtz.append(qtz)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """The forward function of the searchable mixed-precision layer.

        In a nutshell, computes the different quantized representations of `mix_qtz`
        and combine them weighting the different terms channel-wise by means of
        softmax-ed `alpha_prec` trainable parameters.

        :param input: the input float tensor
        :type input: torch.Tensor
        :return: the output fake-quantized with searchable precision tensor
        :rtype: torch.Tensor
        """
        soft_alpha = nn.functional.softmax(self.alpha_prec / self._temperature,
                                           dim=0)
        soft_alpha = soft_alpha.view((self.cout,) + (1,) * len(input.shape[1:]))
        y = []
        for i, quantizer in enumerate(self.mix_qtz):
            y.append(soft_alpha[i] * quantizer(input))
        y = torch.stack(y, dim=0).sum(dim=0)
        return y

    @property
    def temperature(self) -> float:
        """Returns the actual softmax temperature for this layer.

        :return: the actual softmax temperature for this layer.
        :rtype: float
        """
        return self._temperature

    @temperature.setter
    def input_features_calculator(self, tau: float):
        """Set the softmax temperature for this layer.

        :param tau: the softmax temperature to be setted
        :type tau: float
        """
        self._temperature = tau
