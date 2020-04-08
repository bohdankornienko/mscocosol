from mscocosol.dsapproach.object_factory import ObjectFactory

from mscocosol.utils.datagen import make_datagen

datagen_factory = ObjectFactory()

datagen_factory.register_builder('datagen', make_datagen)
