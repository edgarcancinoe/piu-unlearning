from piu_unlearning.config import SISSConfig, PIUConfig, UCEConfig, WIDConfig
from piu_unlearning.methods.siss import SISS
from piu_unlearning.methods.piu import PIU
from piu_unlearning.methods.uce import UCE
from piu_unlearning.methods.wid import WID


def build_method(config: PIUConfig | SISSConfig | UCEConfig | WIDConfig) -> PIU | SISS | UCE | WID:
    if config.method == "uce": return UCE()
    if config.method == "wid": return WID(config.model_loss_weight, config.identity_loss_weight, config.preservation_weight)
    if config.method == "siss": return SISS(config.beta)
    return PIU(config.preservation_weight, config.negative_guidance_scale)
