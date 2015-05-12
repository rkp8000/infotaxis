"""Create a set of geometrical configurations in the database."""

import numpy as np

from db_api.connect import engine, session
from db_api.mappings import Base, GeomConfig, GeomConfigGroup

from config.make_geom_config_group import *


Base.metadata.create_all(engine)

# create geom_config_group
geom_config_group = GeomConfigGroup(id=ID)
geom_config_group.description = DESCRIPTION

# create some geom_configs
geom_configs = []
for _ in range(NGEOMCONFIGS):
    geom_config = GeomConfig()

    geom_config.src_xidx, geom_config.start_xidx = np.random.randint(ENV.nx, size=2)
    geom_config.src_yidx, geom_config.start_yidx = np.random.randint(ENV.ny, size=2)
    geom_config.src_zidx, geom_config.start_zidx = np.random.randint(ENV.nz, size=2)

    geom_config.duration = DURATION

    geom_configs += [geom_config]

geom_config_group.geom_configs += geom_configs

session.add(geom_config_group)
session.commit()