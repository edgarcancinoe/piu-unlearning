from piu_unlearning.config import ESDConfig, PIUConfig, UCEConfig
from piu_unlearning.methods.esd import ESD
from piu_unlearning.methods.piu import PIU
from piu_unlearning.methods.uce import UCE


def build_method(config: PIUConfig | ESDConfig | UCEConfig) -> PIU | ESD | UCE:
    if config.method == "uce": return UCE()
    method_type = {"piu": PIU, "esd": ESD}[config.method]
    return method_type(config.preservation_weight, config.negative_guidance_scale)
