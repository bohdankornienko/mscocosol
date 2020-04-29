from mscocosol.dsapproach.object_factory import ObjectFactory

from mscocosol.approach.models.torch.unet import make_unet
from mscocosol.approach.models.torch.resnet18 import make_resnet18
from mscocosol.approach.models.torch.resnet101 import make_resnet101

from .torch.alexnet import *

model_factory = ObjectFactory()

model_factory.register_builder("unet", make_unet)
model_factory.register_builder("resnet18", make_resnet18)
model_factory.register_builder("resnet101", make_resnet101)

alex_nets = [make_alex_net_v1, make_alex_net_v2, make_alex_net_v3, make_alex_net_v4, make_alex_net_v5 , make_alex_net_v6]
for i, alex_net in enumerate(alex_nets):
    model_factory.register_builder(f"alex_net_v{i+1}", alex_net)
