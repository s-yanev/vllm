# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from transformers import PretrainedConfig


class MimoV2Config(PretrainedConfig):
    model_type = "mimo_v2"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
