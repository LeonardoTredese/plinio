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

from .mixprec_module import MixPrecModule
from .mixprec_identity import MixPrec_Identity
from .mixprec_relu import MixPrec_ReLU
from .mixprec_linear import MixPrec_Linear
from .mixprec_conv2d import MixPrec_Conv2d
from .mixprec_qtz import MixPrecType
from .mixprec_add import MixPrec_Add

__all__ = [
    'MixPrecModule', 'MixPrec_Identity', 'MixPrec_ReLU',
    'MixPrec_Linear', 'MixPrec_Conv2d', 'MixPrecType', 'MixPrec_Add',
]
