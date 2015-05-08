from sqlalchemy import ForeignKey, Column, Sequence, Integer, String
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship, backref

Base = declarative_base()


class Simulation(Base):
    __tablename__ = 'simulation'


class GeomConfigGroup(Base):
    __tablename__ = 'geom_config_group'


class GeomConfig(Base):
    __tablename__ = 'geom_config'


class Trajectory(Base):
    __tablename__ = 'trajectory'


class TrajectoryInfo(Base):
    __tablename__ = 'trajectory_info'


class Insect(Base):
    __tablename__ = 'insect'


class InsectParam(Base):
    __tablename__ = 'insect_param'


class Plume(Base):
    __tablename__ = 'plume'


class PlumeParam(Base):
    __tablename__ = 'plume_param'


class Timepoint(Base):
    __tablename__ = 'timepoint'