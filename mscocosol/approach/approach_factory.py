from mscocosol.dsapproach.object_factory import ObjectFactory

from .torch_based import make_torch_based_approach
from .single_channel import make_single_channel_approach

approach_factory = ObjectFactory()

approach_factory.register_builder("torch_based", make_torch_based_approach)
approach_factory.register_builder("single_channel", make_single_channel_approach)
