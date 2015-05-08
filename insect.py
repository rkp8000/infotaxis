"""
Created on Thu Dec 18 15:28:52 2014

@author: rkp

Contains class for infotaxis-style insect.
"""

import numpy as np
from numpy import concatenate as cc
from itertools import product as cproduct

from math_tools.fun import entropy
from math_tools.array_management import nearest


class Insect(object):
    """Insect class.
    
    Args:
        env: Environment in which insect lives.
    """

    name = 'basic'

    # params that must be set before initializing
    param_names = ('r', 'd', 'a', 'tau', 'w',
                   'loglike_function')

    # potential move array (idxs)
    moves = (np.array([1, 0, 0]),
             np.array([-1, 0, 0]),
             np.array([0, 1, 0]),
             np.array([0, -1, 0]),
             np.array([0, 0, 1]),
             np.array([0, 0, -1]))

    def __init__(self, env, dt=.01):
        """Insect constructor."""
        # store environment
        self.env = env
        self.dt = dt

        # set default parameters
        self.r = None  # estimated source rate
        self.d = None  # estimated diffusion coefficient (m^2 / s)
        self.a = None  # estimated particle radius (m)
        self.tau = None  # particle lifetime (s)
        self.w = None  # estimate of wind speed (m/s)
        self.loglike_function = None  # log-likelihood function
        self.odor_domain = None  # domain of odor possibilities
        
        # set some other variables to their null values
        self.params = {}
        self.next_pos_idxs = None
        self.move_utils = np.zeros((len(self.moves),), dtype=float)
        self.prob = None
        self.logprob = None
        self.logprob = None

    def set_params(self, w=None, r=None, d=None, a=None, tau=None):
        """Set insect's parameters equal to the plume's (i.e., so it has perfect
        knowledge of the plume statistics)."""

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

    def set_pos(self, pos, is_idx=False):
        """Set insect position."""

        if is_idx:
            self.pos_idx = pos
        else:
            # extract idx from position
            self.pos_idx = self.env.idx_from_pos(pos)

        # convert idxs to positions
        self.pos = self.env.pos_from_idx(self.pos_idx)

    def initialize(self):
        """Reset insect history."""

        # check that all params are set
        for param in [self.r, self.d, self.a, self.tau, self.w,
                      self.loglike_function]:
            if param is None:
                param_str = ', '.join(self.param_names)
                msg = ('Please make sure all of the following parameters are set:'
                       '%s.' % param_str)
                raise ValueError(msg)

        # reset clock
        self.ts = 0 # timestep
        self.t = 0. # time
        
        # reset src pos logprob
        self.logprob = np.zeros(self.env.shape, dtype=float)
        
        # get odor domain from loglikelihood function
        self.odor_domain = self.loglike_function.domain

        # calculate extended loglikelihood map

        # get environment geometry
        xext = self.env.x - self.env.x[0]
        xext = cc([-xext[1:][::-1], xext])
        yext = self.env.y - self.env.y[0]
        yext = cc([-yext[1:][::-1], yext])
        zext = self.env.z - self.env.z[0]
        zext = cc([-zext[1:][::-1], zext])

        central_idx = ((len(xext) - 1)/2, (len(yext) - 1)/2, (len(zext) - 1)/2)

        map_shape = (len(xext), len(yext), len(zext))
        self.loglike_map = np.zeros((len(self.odor_domain), ) + map_shape, dtype=float)

        for odor_idx, odor in enumerate(self.odor_domain):
            loglike = self.loglike_function(odor, central_idx, xext=xext,
                                            yext=yext, zext=zext, dt=self.dt,
                                            w=self.w, r=self.r, d=self.d,
                                            a=self.a, tau=self.tau)

            self.loglike_map[odor_idx] = loglike

    def get_loglike(self, odor, pos_idx):
        """Return the relevant portion of the source loglikelihood map (i.e.,
        the probability of the source being at any given location in the
        environment given a specified odor value at a specified position."""

        # get idx of this odor
        odor_idx = nearest(odor, self.odor_domain)

        # get correct slice of loglikelihood map
        xidx, yidx, zidx = pos_idx
        xslice = slice(self.env.nx - 1 - xidx, 2*self.env.nx - 1 - xidx)
        yslice = slice(self.env.ny - 1 - yidx, 2*self.env.ny - 1 - yidx)
        zslice = slice(self.env.nz - 1 - zidx, 2*self.env.nz - 1 - zidx)

        return self.loglike_map[odor_idx][xslice, yslice, zslice]

    def get_src_prob(self, log=True, normalized=False):

        if normalized and not log:
            return self.prob
        elif log and not normalized:
            return self.logprob - self.logprob.max()
        else:
            raise NotImplementedError('Functionality not yet present.')

    @property
    def logprobxy(self):
        return self.logprob[:, :, self.env.center_zidx]

    @property
    def logprobxz(self):
        return self.logprob[:, self.env.center_yidx, :]

    def get_utility_map(self):
        """Calculate utility for moving to every position in environment."""
        next_pos_idxs = list(cproduct(range(self.env.nx),
                                      range(self.env.ny),
                                      range(self.env.nz)))

        self.calc_util(next_pos_idxs)

        return self.move_utils.reshape(self.env.shape)

    def get_next_pos_idxs(self):
        """Get the possible position indices for the next move."""

        self.next_pos_idxs = np.array(self.pos_idx) + self.moves

        return self.next_pos_idxs

    def sample(self, odor):
        """Store internal odor value, setting the binary value hit to True if
        the odor value is greater than zero.
        
        Args:
            odor: odor value sampled from plume; this must be in the insect's
            odor domain"""
            
        self.odor = self.odor_domain[nearest(odor, self.odor_domain)]

        if self.odor > 0:
            self.hit = True
        else:
            self.hit = False
        
        return self.odor
        
    def update_src_prob(self, odor=None, pos_idx=None, store=True, log=True):
        """Update probability distribution over src pos.
        
        Args:
            store: set to True to store updated log source probability
            log: set to True to calculate unnormalized log probability. If 
                False, then normalized probability will be returned (not log)."""
        
        # if no odor or pos_idx provided, assume we're using the stored one
        if odor is None:
            odor = self.odor
        if pos_idx is None:
            pos_idx = self.pos_idx

        # create unnormalized log posterior
        logposterior = self.get_loglike(odor, pos_idx) + self.logprob

        # set probability to zero (log to -inf) at insect's position
        logposterior[tuple(pos_idx)] = -np.inf

        # if all values are -np.inf, set them all to zero, otherwise translate
        # so that max value is 0 to keep things within reasonable dyn. range
        if np.all(np.isinf(logposterior)*(logposterior < 0)):
            logposterior[:] = 0.
        else:
            logposterior -= logposterior.max()

        # either store or return log src probability
        if store:
            self.logprob = logposterior
            # store real probability and entropy
            # calculate normalized probability
            prob = np.exp(self.logprob)
            # normalize
            prob /= prob.sum()
            self.prob = prob
            self.S = entropy(prob)
        else:
            if log:
                return logposterior
            else:
                # calculate normalized probability
                prob = np.exp(logposterior)
                # normalize
                prob /= prob.sum()
                return prob

    def calc_util(self, next_pos_idxs=None):
        """Calculate the utility of moving in any given direction."""
        
        # set and use next pos idxs if they haven't been provided
        if next_pos_idxs is None:
            next_pos_idxs = self.get_next_pos_idxs()
        
        # calculate utility for all moves
        self.move_utils[:] = -np.inf
        
        for move_ctr, next_pos_idx in enumerate(next_pos_idxs):
            
            # leave at minimum if move idx is out of bounds
            if self.env.idx_out_of_bounds(next_pos_idx):
                continue

            # get change in entropy if source found
            deltaS_find_src = -self.S

            # calculate expected change in entropy if source not found

            # calculate probability of all possible odor sample values and
            # change in src pos entropy resulting from each odor
            odor_prob = np.zeros(self.odor_domain.shape, dtype=float)
            deltaS = np.zeros(self.odor_domain.shape, dtype=float)
            
            for odor_idx, odor in enumerate(self.odor_domain):
                # get src prior, src like & src-odor joint
                like = np.exp(self.get_loglike(odor, next_pos_idx))
                joint = like * self.prob
                
                # marginalize over src position to get odor prob
                odor_prob[odor_idx] = np.sum(joint)

                # calculate change in entropy of src pos distribution given
                # this odor
                next_prob = self.update_src_prob(odor, next_pos_idx,
                                                 store=False, log=False)
                deltaS[odor_idx] = entropy(next_prob) - self.S
                
            # calculate expected change in entropy if source not found
            # (averaged over odors)
            exptd_deltaS_no_src = np.dot(deltaS, odor_prob)
            
            ## calculate probability of finding source
            pfind_src = self.prob[tuple(next_pos_idx)]
            pno_src = 1 - pfind_src

            # calculate final expected change in entropy
            exptd_deltaS = exptd_deltaS_no_src*pno_src + deltaS_find_src*pfind_src
            
            if np.isnan(exptd_deltaS):
                print 'Error! Switching to debugging mode...'
                import pdb; pdb.set_trace()
            
            self.move_utils[move_ctr] = -exptd_deltaS
        
    def move(self, next_pos_idx=None):
        """Move the insect by selecting the highest utility move."""
        
        # pick next move if not provided
        if next_pos_idx is None:
            next_pos_idx = self.next_pos_idxs[np.argmax(self.move_utils)]
        
        # store idx and pos
        self.pos_idx = tuple(next_pos_idx)
        self.pos = self.env.pos_from_idx(self.pos_idx)
        
        # update time
        self.ts += 1
        self.t += self.dt