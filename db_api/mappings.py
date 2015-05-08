from sqlalchemy import Column, ForeignKey, Sequence
from sqlalchemy import Boolean, Integer, BigInteger, Float, String, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship, backref

Base = declarative_base()


class Simulation(Base):
    __tablename__ = 'simulation'

    id = Column(String(255), primary_key=True)

    description = Column(Text)
    total_trials = Column(Integer)

    xmin = Column(Float)
    xmax = Column(Float)
    nx = Column(Integer)

    ymin = Column(Float)
    ymax = Column(Float)
    ny = Column(Integer)

    zmin = Column(Float)
    zmax = Column(Float)
    nz = Column(Integer)

    dt = Column(Float)

    geom_config_group_id = Column(String(255), ForeignKey('geom_config_group.id'))

    plume_id = Column(Integer, ForeignKey('plume.id'))
    insect_id = Column(Integer, ForeignKey('insect.id'))

    ongoing_run_id = Column(Integer, ForeignKey('ongoing_run.id'))

    trials = relationship("Trial", backref='simulation')


class OngoingRun(Base):
    __tablename__ = 'ongoing_run'

    id = Column(Integer, primary_key=True)

    trials_completed = Column(Integer)

    simulations = relationship("Simulation", backref='ongoing_run')


class GeomConfigGroup(Base):
    __tablename__ = 'geom_config_group'

    id = Column(String(255), primary_key=True)

    description = Column(Text)

    simulations = relationship("Simulation", backref='geom_config_group')
    geom_configs = relationship("GeomConfig", backref='geom_config_group')


class GeomConfig(Base):
    __tablename__ = 'geom_config'

    id = Column(Integer, primary_key=True)

    src_xidx = Column(Integer)
    src_yidx = Column(Integer)
    src_zidx = Column(Integer)

    start_xidx = Column(Integer)
    start_yidx = Column(Integer)
    start_zidx = Column(Integer)

    duration = Column(Integer)

    geom_config_group_id = Column(String(255), ForeignKey('geom_config_group.id'))

    trials = relationship("Trial", backref='geom_config')


class Trial(Base):
    __tablename__ = 'trial'

    id = Column(Integer, primary_key=True)

    simulation_id = Column(String(255), ForeignKey('simulation.id'))
    geom_config_id = Column(Integer, ForeignKey('geom_config.id'))

    trial_info = relationship("TrialInfo", backref='trial')

    start_timepoint_id = Column(BigInteger)
    end_timepoint_id = Column(BigInteger)


class TrialInfo(Base):
    __tablename__ = 'trial_info'

    id = Column(Integer, primary_key=True)

    trial_id = Column(Integer, ForeignKey('trial.id'))

    duration = Column(Integer)
    found_src = Column(Boolean)


class Insect(Base):
    __tablename__ = 'insect'

    id = Column(Integer, primary_key=True)

    simulations = relationship("Simulation", backref='insect')
    insect_params = relationship("InsectParam", backref='insect')


class InsectParam(Base):
    __tablename__ = 'insect_param'

    id = Column(Integer, primary_key=True)

    name = Column(String(50))
    value = Column(Float)

    insect_id = Column(Integer, ForeignKey('insect.id'))


class Plume(Base):
    __tablename__ = 'plume'

    id = Column(Integer, primary_key=True)

    simulations = relationship("Simulation", backref='plume')
    plume_params = relationship("PlumeParam", backref='plume')


class PlumeParam(Base):
    __tablename__ = 'plume_param'

    id = Column(Integer, primary_key=True)

    name = Column(String(50))
    value = Column(Float)

    plume_id = Column(Integer, ForeignKey('plume.id'))


class Timepoint(Base):
    __tablename__ = 'timepoint'

    id = Column(BigInteger, primary_key=True)

    xidx = Column(Integer)
    yidx = Column(Integer)
    zidx = Column(Integer)

    hxy = Column(Float)
    hxyz = Column(Float)

    hxy_smoothed = Column(Float)
    hxyz_smoothed = Column(Float)