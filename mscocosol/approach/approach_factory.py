from mscocosol.dsapproach.object_factory import ObjectFactory

from mscocosol.approach.unet import make_unet_torch

approach_factory = ObjectFactory()

approach_factory.register_builder("unet_torch", make_unet_torch)