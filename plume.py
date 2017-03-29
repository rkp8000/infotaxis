"""
Created on Thu Dec 18 12:07:31 2014

@author: rkp

Classes for various types of plumes.
"""

import numpy as np
from scipy.stats import multivariate_normal as mvn
from logprob_odor import advec_diff_mean_hit_rate


class Environment3d(object):
    """3D environment object."""

    @staticmethod
    def diagonalest_lattice_path(r0, r1):
        """Return a lattice path between r0 and r1 that is as close to diagonal as possible."""
        dx = r1[0] - r0[0]
        dy = r1[1] - r0[1]
        dz = r1[2] - r0[2]

        xstep = [np.sign(dx), 0, 0]
        ystep = [0, np.sign(dy), 0]
        zstep = [0, 0, np.sign(dz)]

        # to determine which lattice path yields the most "diagonal" path, we split the
        # interval [0, 1] into segments for dx, dy, and dz, and then sort the steps by
        # where they lie on the interval

        xpts = np.linspace(0, 1, np.abs(dx) + 2)[1:-1]
        ypts = np.linspace(0, 1, np.abs(dy) + 2)[1:-1]
        zpts = np.linspace(0, 1, np.abs(dz) + 2)[1:-1]

        unsorted_steps = np.concatenate([np.tile(xstep, (np.abs(dx), 1)),
                                         np.tile(ystep, (np.abs(dy), 1)),
                                         np.tile(zstep, (np.abs(dz), 1))], axis=0)

        sort_keys = np.concatenate([xpts, ypts, zpts])

        # sort all steps
        sorted_steps = unsorted_steps[np.argsort(sort_keys)]

        # get actual position indexs
        pos_idxs = sorted_steps.cumsum(axis=0) + r0

        return pos_idxs

    def __init__(self, xbins, ybins, zbins):
        # store bins
        self.xbins = xbins
        self.ybins = ybins
        self.zbins = zbins

        # calculate bin centers
        self.x = 0.5 * (xbins[:-1] + xbins[1:])
        self.y = 0.5 * (ybins[:-1] + ybins[1:])
        self.z = 0.5 * (zbins[:-1] + zbins[1:])

        # get ranges
        self.xr = self.x[-1] - self.x[0]
        self.yr = self.y[-1] - self.y[0]
        self.zr = self.z[-1] - self.z[0]

        # get index environment
        self.xidx = np.arange(len(self.x), dtype=int)
        self.yidx = np.arange(len(self.y), dtype=int)
        self.zidx = np.arange(len(self.z), dtype=int)

        # store other useful information
        self.dx = xbins[1] - xbins[0]
        self.dy = ybins[1] - ybins[0]
        self.dz = zbins[1] - zbins[0]

        self.nx = len(self.x)
        self.ny = len(self.y)
        self.nz = len(self.z)
        self.shape = (self.nx, self.ny, self.nz)

        # for quickly calculating idxs from positions
        self.xslope = self.xidx[-1]/self.xr
        self.yslope = self.yidx[-1]/self.yr
        if self.nz > 1:
            self.zslope = self.zidx[-1]/self.zr
        else:
            self.zslope = 0

        self.xint = self.x[0]
        self.yint = self.y[0]
        self.zint = self.z[0]

        # get center idxs
        self.center_xidx = int(np.floor(self.nx/2))
        self.center_yidx = int(np.floor(self.ny/2))
        self.center_zidx = int(np.floor(self.nz/2))

        # make extents for plotting
        self.extentxy = [self.xbins[0], self.xbins[-1], self.ybins[0], self.ybins[-1]]
        self.extentxz = [self.xbins[0], self.xbins[-1], self.zbins[0], self.zbins[-1]]
        self.extentyz = [self.ybins[0], self.ybins[-1], self.zbins[0], self.zbins[-1]]

    def pos_from_idx(self, idx):
        """Get floating point position from index."""

        return self.x[idx[0]], self.y[idx[1]], self.z[idx[2]]

    def idx_from_pos(self, pos):
        """Return the index corresponding to the specified postion."""

        xidx = np.round((pos[0] - self.xint) * self.xslope)
        yidx = np.round((pos[1] - self.yint) * self.yslope)
        zidx = np.round((pos[2] - self.zint) * self.zslope)

        xidx = int(xidx)
        yidx = int(yidx)
        zidx = int(zidx)

        if xidx < 0:
            xidx = 0
        elif xidx >= self.nx:
            xidx = self.nx - 1

        if yidx < 0:
            yidx = 0
        elif yidx >= self.ny:
            yidx = self.ny - 1

        if zidx < 0:
            zidx = 0
        elif zidx >= self.nz:
            zidx = self.nz - 1

        return xidx, yidx, zidx

    def idx_out_of_bounds(self, pos_idx):
        """Check whether position idx is out of bounds."""
        if np.any(np.less(pos_idx, 0)):
            return True
        elif np.any(np.greater_equal(pos_idx, (self.nx, self.ny, self.nz))):
            return True

        return False

    def discretize_position_sequence(self, positions):

        # convert all positions to idxs
        idxs = np.array([self.idx_from_pos(pos) for pos in positions])

        # calculate L1 distance between every pair of adjacent idxs
        l1_dists = np.abs(np.diff(idxs, axis=0)).sum(axis=1)

        # split idxs into chunks such of adjacent idxs
        split_idxs = (l1_dists > 1).nonzero()[0]
        chunks = np.split(idxs, split_idxs + 1)

        # calculate connecting path for each pair of adjacent chunks
        connectors = []
        for cctr, chunk in enumerate(chunks[:-1]):
            next_chunk = chunks[cctr + 1]
            connectors += [self.diagonalest_lattice_path(chunk[-1], next_chunk[0])]

        # interleave chunks with connectors
        temp_idxs = [None] * (len(chunks) + len(connectors))
        temp_idxs[::2] = chunks
        temp_idxs[1::2] = connectors
        temp_idxs = np.concatenate(temp_idxs)

        # remove adjacent duplicates
        final_idxs = [temp_idxs[0]]
        for temp_idx in temp_idxs[1:]:
            if not np.all(temp_idx == final_idxs[-1]):
                final_idxs += [temp_idx]

        return final_idxs


class Plume(object):
    
    def __init__(self, env, dt=.01, orm=None):
        
        # set bins and timestep
        self.env = env
        self.dt = dt

        self.ts = 0
        self.t = 0.

        # get dimensionality of plume
        if self.env.nz == 1:
            self.dim = 2
        else:
            self.dim = 3

        # set all variables to none
        self.params = {}

        self.src_pos = None
        self.src_pos_idx = None
        self.srcx = None
        self.srcy = None
        self.srcz = None

        self.srcxidx = None
        self.srcyidx = None
        self.srczidx = None

        self.conc = None

        self._orm = None

        if orm:
            self.orm = orm

    def reset(self):
        """Reset plume params."""
        # reset time and timestep
        self.ts = 0
        self.t = 0.
    
    def set_src_pos(self, pos, is_idx=False):
        """Set source position."""

        if is_idx:
            xidx, yidx, zidx = pos
        else:
            xidx, yidx, zidx = self.env.idx_from_pos(pos)
            
        # store src position
        self.src_pos_idx = (xidx, yidx, zidx)
        # convert idxs to positions
        self.src_pos = self.env.pos_from_idx(self.src_pos_idx)

        # create some other useful variables
        self.srcxidx, self.srcyidx, self.srczidx = self.src_pos_idx
        self.srcx, self.srcy, self.srcz = self.src_pos
                
    def update_time(self):
        """Update time."""
        self.ts += 1
        self.t += self.dt
        
    def update(self):
        """Update everything."""
        self.update_time()

    @property
    def concxy(self):
        return self.conc[:, :, self.env.center_zidx]

    @property
    def concxz(self):
        return self.conc[:, self.env.center_yidx, :]

    @property
    def concyz(self):
        return self.conc[self.env.center_xidx, :, :]

    def generate_orm(self, models, sim=None):
        """Set up the object relational mapping of the plume.

        The models object must have as an attribute a model called Plume."""

        self._orm = models.Plume(type=self.name)
        self._orm.plume_params = [models.PlumeParam(name=n, value=v) for n, v in self.params.items()]

        if sim:
            self._orm.simulations = [sim]

    @property
    def orm(self):
        return self._orm

    @orm.setter
    def orm(self, orm):
        self._orm = orm
        param_dict = {pp.name: pp.value for pp in self._orm.plume_params}
        self.set_params(**param_dict)


class EmptyPlume(Plume):
    
    name = 'empty'
    
    def initialize(self):
        
        # create meshgrid arrays for setting conc
        x, y, z = np.meshgrid(self.env.x, self.env.y, self.env.z, indexing='ij')
        
        # create empty conc plume
        self.conc = np.zeros(x.shape, dtype=float)
        
        # store odor domain
        self.odor_domain = [0, 1]
    
    def sample(self, pos_idx):
        return 0


class CollimatedPlume(Plume):

    name = 'collimated'

    def set_params(self, max_conc=None, threshold=None, ymean=None, zmean=None, ystd=None, zstd=None):
        """params of real plume:
            ymean = 0.0105
            zmean = 0.0213

            ystd = .0073
            zstd = .0094

            max_conc = 488
        """
        if max_conc is not None:
            self.params['max_conc'] = max_conc
            self.max_conc = max_conc

        if threshold is not None:
            self.params['threshold'] = threshold
            self.threshold = threshold

        if ymean is not None:
            self.params['ymean'] = ymean
            self.ymean = ymean
        if zmean is not None:
            self.params['zmean'] = zmean
            self.zmean = zmean

        if ystd is not None:
            self.params['ystd'] = ystd
            self.ystd = ystd
        if zstd is not None:
            self.params['zstd'] = zstd
            self.zstd = zstd

    def initialize(self):
        # create meshgrid of all locations
        x, y, z = np.meshgrid(self.env.x, self.env.y, self.env.z, indexing='ij')

        cov = np.array([[self.ystd**2, 0], [0, self.zstd**2]])

        exponent = (-0.5 * ((y - self.ymean)**2) / (self.ystd**2)) + (-0.5 * ((z - self.zmean)**2) / (self.zstd**2))
        self.conc = self.max_conc * np.exp(exponent)

    def sample(self, pos_idx):
        if self.conc[tuple(pos_idx)] > self.threshold >= 0:
            return 1
        else:
            return 0


class SpreadingGaussianPlume(Plume):
    """
    Plume whose cross-section is always Gaussian & whose magnitude decreases hyperbolically (as 1/x) with distance from the source. This is based on "biology and the mechanics of the wave-swept environment" by Mark Denny (pp. 144-147).

    The parameters and equation are from Floris van Breugel (https://github.com/florisvb/DataFit).
    """

    name = 'spreading_gaussian'

    def set_params(self, **kwargs):
        """
        Set parameters
        :param Q: scaling factor (default -0.26618286981003886)
        :param u: wind speed (default 0.4)
        :param u_star: some parameter (default 0.06745668765535813)
        :param alpha_y: some parameter (default -0.066842568000323691)
        :param alpha_z: some parameter (default 0.14538827993452938)
        :param x_source: x source position (default -0.64790143304753445)
        :param y_source: y source position (default .003)
        :param z_source: z source position (default .011)
        :param bkgd: background level (default 400)
        :param threshold: threshold for detection of plume (default 450)
        """

        for k, v in kwargs.items():
            self.params[k] = v
            self.__dict__[k] = v

    def initialize(self):
        # create meshgrid of all locations
        x, y, z = np.meshgrid(self.env.x, self.env.y, self.env.z, indexing='ij')

        y_term = ((y - self.y_source)**2) * (self.u**2)
        y_term /= (2 * (self.alpha_y**2) * (self.u_star**2) * ((x - self.x_source)**2))

        z_term = ((z - self.z_source)**2) * (self.u**2)
        z_term /= (2 * (self.alpha_z**2) * (self.u_star**2) * ((x - self.x_source)**2))

        c = (self.Q * self.u)
        c /= (2 * np.pi * self.alpha_y * self.alpha_z * (self.u_star**2) * (x - self.x_source)**2)
        c *= np.exp(-(y_term + z_term))
        c += self.bkgd

        self.conc = c

    def sample(self, pos_idx):
        if self.conc[tuple(pos_idx)] > self.threshold >= 0:
            return 1
        else:
            return 0


class PoissonPlume(Plume):
    """In Poisson Plumes, odor samples are given by draws from a Poisson 
    distribution with mean equal to concentration times timestep (dt)"""

    max_hit_number = 1

    def sample(self, pos_idx):
        # get concentration
        conc = self.conc[tuple(pos_idx)]
        
        # sample odor from concentration, capping it at max hit number
        odor = min(np.random.poisson(lam=conc*self.dt), self.max_hit_number)
        
        return odor
        

class BasicPlume(PoissonPlume):
    """Stationary advection-diffusion based plume.
    Specified by source rate, diffusivity, particle size, and decay time, wind,
    and integration time.

    Args:
        w: wind speed (wind blows from negative to positive x-direction) (m/s)
        R: source emission rate
        D: diffusivity (m^2/s)
        a: searcher size (m)
        tau: particle lifetime (s)

    Default params: w: 0.4
                    r: 10
                    d: 0.1
                    a: .002
                    tau: 1000
        """

    name = 'basic'

    def set_params(self, w=None, r=None, d=None, a=None, tau=None):
        # store auxiliary parameters
        if w:
            self.params['w'] = w
            self.w = w
        if r:
            self.params['r'] = r
            self.r = r
        if d:
            self.params['d'] = d
            self.d = d
        if a:
            self.params['a'] = a
            self.a = a
        if tau:
            self.params['tau'] = tau
            self.tau = tau

    def initialize(self):
        # create meshgrid of all locations
        x, y, z = np.meshgrid(self.env.x, self.env.y, self.env.z, indexing='ij')
        # calculate displacement from source
        dx = x - self.src_pos[0]
        dy = y - self.src_pos[1]
        dz = z - self.src_pos[2]

        # calculate mean hit number at all locations
        self.mean_hit_rate = advec_diff_mean_hit_rate(dx, dy, dz,
                                                      self.w, self.r, self.d,
                                                      self.a, self.tau, self.dim)
        self.conc = self.mean_hit_rate

        # store odor domain
        self.odor_domain = range(self.max_hit_number+1)

    def sample(self, pos_idx, dt=None):
        if not dt:
            dt = self.dt
        # randomly sample from plume
        mean_hit_num = self.mean_hit_rate[tuple(pos_idx)] * dt
        if not np.isinf(mean_hit_num):
            hit_num = np.random.poisson(lam=mean_hit_num)
        else:
            hit_num = np.inf

        return 0  #min(hit_num, self.max_hit_number)


class CollimatedPoissonPlume(PoissonPlume):
    """Stationary collimated plume. Specified by width (meters) and
    peak concentration."""
    
    name = 'collimated_poisson'
    
    def set_aux_params(self, width, peak, max_hit_number):
        # store auxiliary parameters
        self.width = width
        self.peak = peak
        self.max_hit_number = int(max_hit_number)
    
    def initialize(self):

        # create meshgrid arrays for setting conc
        x, y, z = np.meshgrid(self.env.x, self.env.y, self.env.z, indexing='ij')
        
        # calculate conc concentration
        dr2 = (y - self.src_pos[1])**2 + (z - self.src_pos[2])**2
        
        self.conc = self.peak * np.exp(-dr2 / (2*self.width))
        
        # put mask over space upwind of src
        mask = (x < self.src_pos[0])
        self.conc[mask] = 0.
        
        # store odor domain
        self.odor_domain = range(self.max_hit_number+1)