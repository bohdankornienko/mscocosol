from mscocosol.dsapproach.object_factory import ObjectFactory

from mscocosol.approach.torch_based import make_torch_based_approach

approach_factory = ObjectFactory()

approach_factory.register_builder("torch_based", make_torch_based_approach)
