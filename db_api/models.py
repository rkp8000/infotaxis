import numpy as np
from sqlalchemy import Column, ForeignKey, Sequence
from sqlalchemy import Boolean, Integer, BigInteger, Float, String, Text, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship, backref

from plume import Environment3d

from connect import engine

Base = declarative_base()


class Simulation(Base):
    __tablename__ = 'simulation'
    _env = None

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

    heading_smoothing = Column(Integer)

    geom_config_group_id = Column(String(255), ForeignKey('geom_config_group.id'))

    plume_id = Column(Integer, ForeignKey('plume.id'))
    insect_id = Column(Integer, ForeignKey('insect.id'))

    ongoing_run_id = Column(Integer, ForeignKey('ongoing_run.id'))

    trials = relationship("Trial", backref='simulation')

    @property
    def env(self):
        if not self._env:
            xbins = np.linspace(self.xmin, self.xmax, self.nx + 1)
            ybins = np.linspace(self.ymin, self.ymax, self.ny + 1)
            zbins = np.linspace(self.zmin, self.zmax, self.nz + 1)
            self._env = Environment3d(xbins, ybins, zbins)
        return self._env

    @env.setter
    def env(self, env):
        self._env = env

        self.xmin = env.xbins[0]
        self.xmax = env.xbins[-1]
        self.nx = env.nx

        self.ymin = env.ybins[0]
        self.ymax = env.ybins[-1]
        self.ny = env.ny

        self.zmin = env.zbins[0]
        self.zmax = env.zbins[-1]
        self.nz = env.nz


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

    @property
    def src_idx(self):
        return self.src_xidx, self.src_yidx, self.src_zidx

    @property
    def start_idx(self):
        return self.start_xidx, self.start_yidx, self.start_zidx


class Trial(Base):
    __tablename__ = 'trial'

    id = Column(Integer, primary_key=True)

    simulation_id = Column(String(255), ForeignKey('simulation.id'))
    geom_config_id = Column(Integer, ForeignKey('geom_config.id'))

    trial_info = relationship("TrialInfo", uselist=False, backref='trial')

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
    type = Column(String(255))

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
    type = Column(String(255))

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

    hxyz = Column(Float)

    odor = Column(Float)
    detected_odor = Column(Float)

    src_entropy = Column(Float)


class Script(Base):
    __tablename__ = 'script'

    id = Column(String(255), primary_key=True)

    description = Column(Text)
    type = Column(String(255))

    script_executions = relationship("ScriptExecution", backref='script')


class ScriptExecution(Base):
    __tablename__ = 'script_execution'

    id = Column(Integer, primary_key=True)

    script_id = Column(String(255), ForeignKey('script.id'))
    commit = Column(String(255))
    timestamp = Column(DateTime)
    notes = Column(Text)


class GeomConfigExtensionRealTrajectory(Base):
    __tablename__ = 'geom_config_extension_real_trajectory'

    id = Column(Integer, primary_key=True)

    avg_dt = Column(Float)

    geom_config_id = Column(Integer, ForeignKey('geom_config.id'))
    trial_id = Column(Integer, ForeignKey('trial.id'))
    real_trajectory_id = Column(String(255))

    geom_config = relationship("GeomConfig", backref=backref('geom_config_extension_real_trajectory', uselist=False))
    trial = relationship("Trial", backref=backref('geom_config_extension_real_trajectory', uselist=False))


if __name__ == '__main__':
    Base.metadata.create_all(engine)