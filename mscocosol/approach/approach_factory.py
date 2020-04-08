from mscocosol.dsapproach.object_factory import ObjectFactory

from mscocosol.approach.unet import make_unet

approach_factory = ObjectFactory()

approach_factory.register_builder("unet", make_unet)
