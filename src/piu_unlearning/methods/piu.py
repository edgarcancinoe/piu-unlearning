from piu_unlearning.methods.reference import proximity_reference
from piu_unlearning.training.losses import GuidedNoiseLoss


class PIU(GuidedNoiseLoss):
    prepare_reference = staticmethod(proximity_reference)
