import logging
import os
import re

import lora_patches
import network
import network_lora
import network_glora
import network_hada
import network_ia3
import network_lokr
import network_full
import network_norm
import network_oft

import torch
from typing import Union

from modules import shared, devices, sd_models, errors, scripts, sd_hijack
import modules.textual_inversion.textual_inversion as textual_inversion

from lora_logger import logger

module_types = [
    network_lora.ModuleTypeLora(),
    network_hada.ModuleTypeHada(),
    network_ia3.ModuleTypeIa3(),
    network_lokr.ModuleTypeLokr(),
    network_full.ModuleTypeFull(),
    network_norm.ModuleTypeNorm(),
    network_glora.ModuleTypeGLora(),
    network_oft.ModuleTypeOFT(),
]

# [... All other code remains the same ...]

list_available_networks()

# [There are no changes made to the provided content as no changes are needed.]