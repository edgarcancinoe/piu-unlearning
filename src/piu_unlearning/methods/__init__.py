from piu_unlearning.config import ESDConfig, PIUConfig, UCEConfig, WIDConfig
from piu_unlearning.methods.esd import ESD
from piu_unlearning.methods.piu import PIU
from piu_unlearning.methods.uce import UCE
from piu_unlearning.methods.wid import WID


def build_method(config: PIUConfig | ESDConfig | UCEConfig | WIDConfig) -> PIU | ESD | UCE | WID:
    if config.method == "uce": return UCE()
    if config.method == "wid": return WID(config.model_loss_weight, config.identity_loss_weight, config.preservation_weight)
    method_type = {"piu": PIU, "esd": ESD}[config.method]
    return method_type(config.preservation_weight, config.negative_guidance_scale)
