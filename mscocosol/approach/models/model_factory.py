from mscocosol.dsapproach.object_factory import ObjectFactory

from mscocosol.approach.models.torch.unet import make_unet
from mscocosol.approach.models.torch.resnet18 import make_resnet18
from mscocosol.approach.models.torch.resnet101 import make_resnet101

model_factory = ObjectFactory()

model_factory.register_builder("unet", make_unet)
model_factory.register_builder("resnet18", make_resnet18)
model_factory.register_builder("resnet101", make_resnet101)
