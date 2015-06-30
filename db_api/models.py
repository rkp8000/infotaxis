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

    @src_idx.setter
    def src_idx(self, src_idx):
        self.src_xidx, self.src_yidx, self.src_zidx = src_idx

    @property
    def start_idx(self):
        return self.start_xidx, self.start_yidx, self.start_zidx

    @start_idx.setter
    def start_idx(self, start_idx):
        self.start_xidx, self.start_yidx, self.start_zidx = start_idx


class Trial(Base):
    __tablename__ = 'trial'

    id = Column(Integer, primary_key=True)

    simulation_id = Column(String(255), ForeignKey('simulation.id'))
    geom_config_id = Column(Integer, ForeignKey('geom_config.id'))

    trial_info = relationship("TrialInfo", uselist=False, backref='trial')

    start_timepoint_id = Column(BigInteger)
    end_timepoint_id = Column(BigInteger)

    def get_timepoints(self, session):
        """Return all timepoints for this trial."""

        timepoints = session.query(Timepoint). \
            filter(Timepoint.id.between(self.start_timepoint_id, self.end_timepoint_id)). \
            order_by(Timepoint.id)

        return timepoints


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


class SegmentGroup(Base):
    __tablename__ = 'segment_group'

    id = Column(String(255), primary_key=True)

    heading_smoothing = Column(Integer)
    threshold = Column(Float)

    simulation_id = Column(String(255), ForeignKey('simulation.id'))

    simulation = relationship("Simulation", backref='segment_groups')


class Segment(Base):
    __tablename__ = 'segment'

    id = Column(Integer, primary_key=True)

    timepoint_id_start = Column(BigInteger)
    timepoint_id_enter = Column(BigInteger)
    timepoint_id_exit = Column(BigInteger)
    timepoint_id_end = Column(BigInteger)

    encounter_number = Column(Integer)

    heading_enter = Column(Float)
    heading_exit = Column(Float)

    x_idx_enter = Column(Integer)
    x_idx_exit = Column(Integer)

    trial_id = Column(Integer, ForeignKey('trial.id'))
    segment_group_id = Column(String(255), ForeignKey('segment_group.id'))

    segment_group = relationship("SegmentGroup", backref='segments')
    trial = relationship("Trial", backref='segments')


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
    real_trajectory_id = Column(String(255))

    geom_config = relationship("GeomConfig", backref=backref('extension_real_trajectory', uselist=False))


################################
# data models used in analysis
class DataArrayInt(Base):
    __tablename__ = 'data_array_int'

    id = Column(Integer, primary_key=True)
    value = Column(Integer)


class SimulationAnalysisPositionHistogram(Base):
    __tablename__ = 'simulation_analysis_position_histogram'

    id = Column(Integer, primary_key=True)

    data_array_start = Column(Integer)
    data_array_end = Column(Integer)

    simulation_id = Column(String(255), ForeignKey('simulation.id'))

    simulation = relationship("Simulation", backref=backref('analysis_position_histogram', uselist=False))

    _data = None

    def store_data(self, session, data):

        # Add all data points to database in order
        data_flat = data.flatten()
        for d_ctr, datum in enumerate(data_flat):
            data_array_int = DataArrayInt(value=datum)
            session.add(data_array_int)

            if d_ctr == 0:
                # calculate start and end ids for the value table
                session.flush()
                self.data_array_start = data_array_int.id
                self.data_array_end = self.data_array_start + len(data_flat) - 1

    def fetch_data(self, session):
        data_flat = session.query(DataArrayInt.value). \
            filter(DataArrayInt.id.between(self.data_array_start, self.data_array_end)). \
            order_by(DataArrayInt.id).all()
        shape = (self.simulation.env.nx,
                 self.simulation.env.ny,
                 self.simulation.env.nz,)

        self._data = np.array(data_flat).reshape(shape)

    @property
    def xy(self):
        if self._data is None:
            raise LookupError('Please run "fetch_data" first!')

        return self._data.sum(axis=2)

    @property
    def xz(self):
        if self._data is None:
            raise LookupError('Please run "fetch_data" first!')

        return self._data.sum(axis=1)

    @property
    def yz(self):
        if self._data is None:
            raise LookupError('Please run "fetch_data" first!')

        return self._data.sum(axis=0)


class SimulationAnalysisDisplacementTotalHistogram(Base):
    __tablename__ = 'simulation_analysis_displacement_total_histogram'

    id = Column(Integer, primary_key=True)

    data_array_start = Column(Integer)
    data_array_end = Column(Integer)

    simulation_id = Column(String(255), ForeignKey('simulation.id'))

    simulation = relationship("Simulation", backref=backref('analysis_displacement_total_histogram', uselist=False))

    _data = None

    def store_data(self, session, data):

        # Add all data points to database in order
        data_flat = data.flatten()
        for d_ctr, datum in enumerate(data_flat):
            data_array_int = DataArrayInt(value=datum)
            session.add(data_array_int)

            if d_ctr == 0:
                # calculate start and end ids for the value table
                session.flush()
                self.data_array_start = data_array_int.id
                self.data_array_end = self.data_array_start + len(data_flat) - 1

    def fetch_data(self, session):
        data_flat = session.query(DataArrayInt.value). \
            filter(DataArrayInt.id.between(self.data_array_start, self.data_array_end)). \
            order_by(DataArrayInt.id).all()
        shape = (2 * self.simulation.env.nx - 1,
                 2 * self.simulation.env.ny - 1,
                 2 * self.simulation.env.nz - 1,)

        self._data = np.array(data_flat).reshape(shape)

    @property
    def extent_xy(self):
        x_max = self.simulation.env.x[-1] - self.simulation.env.dx
        x_min = -x_max
        y_max = self.simulation.env.y[-1] - self.simulation.env.dy
        y_min = -y_max

        return [x_min, x_max, y_min, y_max]

    @property
    def extent_xz(self):
        x_max = self.simulation.env.x[-1] - self.simulation.env.dx
        x_min = -x_max
        z_max = self.simulation.env.z[-1] - self.simulation.env.dz
        z_min = -z_max

        return [x_min, x_max, z_min, z_max]

    @property
    def extent_yz(self):
        y_max = self.simulation.env.y[-1] - self.simulation.env.dy
        y_min = -y_max
        z_max = self.simulation.env.z[-1] - self.simulation.env.dz
        z_min = -z_max

        return [y_min, y_max, z_min, z_max]

    @property
    def xy(self):
        if self._data is None:
            raise LookupError('Please run "fetch_data" first!')

        return self._data.sum(axis=2)

    @property
    def xz(self):
        if self._data is None:
            raise LookupError('Please run "fetch_data" first!')

        return self._data.sum(axis=1)

    @property
    def yz(self):
        if self._data is None:
            raise LookupError('Please run "fetch_data" first!')

        return self._data.sum(axis=0)


class SimulationAnalysisDisplacementAfterNTimestepsHistogram(Base):
    __tablename__ = 'simulation_analysis_displacement_timesteps_histogram'

    id = Column(Integer, primary_key=True)

    data_array_start = Column(Integer)
    data_array_end = Column(Integer)
    n_timesteps = Column(Integer)
    nx = Column(Integer)
    ny = Column(Integer)
    nz = Column(Integer)

    simulation_id = Column(String(255), ForeignKey('simulation.id'))

    simulation = relationship("Simulation",
                              backref='analysis_displacement_after_n_timesteps_histograms')

    _data = None

    def store_data(self, session, data):

        # Add all data points to database in order
        data_flat = data.flatten()
        for d_ctr, datum in enumerate(data_flat):
            data_array_int = DataArrayInt(value=datum)
            session.add(data_array_int)

            if d_ctr == 0:
                # calculate start and end ids for the value table
                session.flush()
                self.data_array_start = data_array_int.id
                self.data_array_end = self.data_array_start + len(data_flat) - 1

    def fetch_data(self, session):
        data_flat = session.query(DataArrayInt.value). \
            filter(DataArrayInt.id.between(self.data_array_start, self.data_array_end)). \
            order_by(DataArrayInt.id).all()

        self._data = np.array(data_flat).reshape((self.nx, self.ny, self.nz,))

    @property
    def shape(self):
        return self.nx, self.ny, self.nz

    @shape.setter
    def shape(self, shape):
        self.nx, self.ny, self.nz = shape

    @property
    def extent_xy(self):
        x_max = self.simulation.env.x[-1] - self.simulation.env.dx
        x_min = -x_max
        y_max = self.simulation.env.y[-1] - self.simulation.env.dy
        y_min = -y_max

        return [x_min, x_max, y_min, y_max]

    @property
    def extent_xz(self):
        x_max = self.simulation.env.x[-1] - self.simulation.env.dx
        x_min = -x_max
        z_max = self.simulation.env.z[-1] - self.simulation.env.dz
        z_min = -z_max

        return [x_min, x_max, z_min, z_max]

    @property
    def extent_yz(self):
        y_max = self.simulation.env.y[-1] - self.simulation.env.dy
        y_min = -y_max
        z_max = self.simulation.env.z[-1] - self.simulation.env.dz
        z_min = -z_max

        return [y_min, y_max, z_min, z_max]

    @property
    def xy(self):
        if self._data is None:
            raise LookupError('Please run "fetch_data" first!')

        return self._data.sum(axis=2)

    @property
    def xz(self):
        if self._data is None:
            raise LookupError('Please run "fetch_data" first!')

        return self._data.sum(axis=1)

    @property
    def yz(self):
        if self._data is None:
            raise LookupError('Please run "fetch_data" first!')

        return self._data.sum(axis=0)


class SimulationAnalysisTakeOffPositionHistogram(Base):
    __tablename__ = 'simulation_analysis_take_off_position_histogram'

    id = Column(Integer, primary_key=True)

    data_array_start = Column(Integer)
    data_array_end = Column(Integer)

    simulation_id = Column(String(255), ForeignKey('simulation.id'))

    simulation = relationship("Simulation", backref=backref('analysis_take_off_position_histogram', uselist=False))

    _data = None

    def store_data(self, session, data):

        # Add all data points to database in order
        data_flat = data.flatten()
        for d_ctr, datum in enumerate(data_flat):
            data_array_int = DataArrayInt(value=datum)
            session.add(data_array_int)

            if d_ctr == 0:
                # calculate start and end ids for the value table
                session.flush()
                self.data_array_start = data_array_int.id
                self.data_array_end = self.data_array_start + len(data_flat) - 1

    def fetch_data(self, session):
        data_flat = session.query(DataArrayInt.value). \
            filter(DataArrayInt.id.between(self.data_array_start, self.data_array_end)). \
            order_by(DataArrayInt.id).all()
        shape = (self.simulation.env.nx,
                 self.simulation.env.ny,
                 self.simulation.env.nz,)

        self._data = np.array(data_flat).reshape(shape)

    @property
    def xy(self):
        if self._data is None:
            raise LookupError('Please run "fetch_data" first!')

        return self._data.sum(axis=2)

    @property
    def xz(self):
        if self._data is None:
            raise LookupError('Please run "fetch_data" first!')

        return self._data.sum(axis=1)

    @property
    def yz(self):
        if self._data is None:
            raise LookupError('Please run "fetch_data" first!')

        return self._data.sum(axis=0)

if __name__ == '__main__':
    Base.metadata.create_all(engine)