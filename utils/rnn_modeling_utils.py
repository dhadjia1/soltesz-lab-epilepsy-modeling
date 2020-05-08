from __future__ import print_function
import numpy as np
from scipy.sparse import csr_matrix
from scipy.linalg.blas import sgemv
from scipy.linalg.blas import sgemm
from functools import partial
from copy import deepcopy
import logging
import time as clock
#from mpi4py import MPI
#from futils import RunningdJ

# def combine_dJ(a,br,idxs):
#     combined = RunningdJ.combine(a,br,idxs)
#     return combined
# mpi_op_combine_dJ = MPI.Op.Create(combined_J, commute=True)


def save_model_data(save_directory, save_prefix, model):
    if not model._init:
        raise Exception('Model was never initialized..')
       
    J = model._J    
    curr_iter = model._curr_iter
    
    train_error      = model._training_error_history[-1]
    validation_error = model._validation_error_history[-1]

    filepath = save_directory + '/' + save_prefix + '-%i.npz' % curr_iter
    f = np.savez(filepath, J=J, train_error=train_error, validation_error=validation_error, curr_iter=curr_iter)



class RNN_modeling(object):
    
    def __init__(self, times, Ytruth, dt, S=None, pc=0.10, tau=2.5, g=1.25, alpha=1.0, float_dtype='float32', comm=None):
        if len(times) != Ytruth.shape[1]:
            raise Exception('time array and Y matrix axis 1 do not align')
           
        self.comm = comm
        if self.comm is not None:
            self.size = comm.Get_size()
            self.rank = comm.Get_rank() 
            N = Ytruth.shape[0]
            self._responsible_nids = []
            for i in range(self.rank, N, self.size):
                self._responsible_nids.append(i)
        else:
            self.rank, self.size = 0, 1
            self._responsible_nids = range(N)
        
        if self.rank == 0:
            self._Ytruth = Ytruth.astype(float_dtype)
           
            self._training_error_history   = []
            self._validation_error_history = []

            self._N  = Ytruth.shape[0]
            self._T  = Ytruth.shape[1]
            if S is not None:
                self._init_S(S.astype(float_dtype))                        
            else: 
                self._S = np.ones((self._N, self._N)).astype(float_dtype)

            self._pc    = pc
            self._tau   = tau
            self._g     = g

            self._J = None
            self._H = None
        else:
            self._N = None
            self.r  = None
            self.e  = None
 
        self._N         = self.comm.bcast(self._N, root=0)           
        self.all_nids   = self.comm.gather(self._responsible_nids, root=0)
        self.sendcounts = self.comm.gather(len(self._responsible_nids) * self._N, root=0)
        self._dt        = dt
        self._times     = times
        self._float_dtype = float_dtype
        self._alpha       = alpha
        self._Pt          = None
        self._W_plastic   = None
        self._connectivity_mask = None
        self._init              = False

        
    def build_from_scratch(self, training_percent=0.80, tvseed=10, jseed=100, ntseed=1000, cmseed=10000, y0seed=100000, y0_range=1.0, noise_trace_std=1.):
   
        if self.rank == 0:
            jrnd    = np.random.RandomState(jseed)
            self._J = jrnd.normal(0., 1., size=(self._N, self._N)).astype(self._float_dtype) * np.float32(self._g/np.sqrt(float(self._N)))
            self._J[np.diag_indices(self._N)] = 0
            self._build(training_percent, tvseed, ntseed, cmseed, y0seed, y0_range, noise_trace_std)
            self._evaluate_model()

        self._init_P(first=True)
        self._curr_iter = 0
        self._init = True
        
    def hot_init(self, J, training_percent=0.80, tvseed=100, ntseed=1000, cmseed=10000, y0seed=100000, y0_range=1.0, noise_trace_std=1.0, curr_iter=0):
        
        if self.rank == 0:
            self._J = J
            self._build(training_percent, tvseed, ntseed, cmseed, y0seed, y0_range, noise_trace_std)
            self._evaluate_model()
                    
        self._init_P(first=True)
        self._curr_iter = curr_iter
        self._init = True
        
    def _build(self, training_percent, tvseed, ntseed, cmseed, y0seed, y0_range, noise_trace_std):

            self.ntrnd   = np.random.RandomState(ntseed)
            self._noise_trace_std   = noise_trace_std    
            self._H = self.ntrnd.normal(0., self._noise_trace_std, size=(self._N, len(self._times))).astype(self._float_dtype)       

            if self._pc < 1.:
                cm_rnd = np.random.RandomState(seed=cmseed)
                self._connectivity_mask = cm_rnd.binomial(1, self._pc, (self._N, self._N)).astype('uint8')
            elif self._pc == 1.:
                self._connectivity_mask = np.ones((self._N, self._N)).astype('uint8')
            else:
                raise Exception('pc must be 0 < x <= 1')
  
            self._connectivity_mask[np.diag_indices(self._N)] = 0
            self._J *= self._connectivity_mask

            self.y0rnd = np.random.RandomState(y0seed)
            self._y0_range = y0_range
            #self._y0 = self.y0rnd.uniform(-self._y0_range, self._y0_range, size=(self._N,)).astype(self._float_dtype)
            self._y0 = self.y0rnd.normal(0., self._y0_range, size=(self._N,)).astype(self._float_dtype)

            
    def _init_S(self, S):
        self._S      = S  # structural connectome          
        min_nonzero  = np.min(self._S[self._S > 0])
        self._S = np.log10(self._S + min_nonzero)
        maxS, minS = np.max(self._S), np.min(self._S)
        self._S    = ((self._S - minS) / (maxS - minS))
        self._S += min_nonzero

    def _init_P(self, first=False):
        if first:
            self._connectivity_mask = self.comm.bcast(self._connectivity_mask, root=0)
            self._W_plastic = [np.asarray(np.nonzero(self._connectivity_mask[i, :])[0], dtype='uint16') for i in self._responsible_nids]
            if self.rank == 0:
                self._W_plastic_full = [np.asarray(np.nonzero(self._connectivity_mask[i, :])[0], dtype='uint16') for i in range(self._N)]
            if self.rank > 0:
                del self._connectivity_mask
        self._Pt = [1./self._alpha*np.identity(len(self._W_plastic[i])).astype(self._float_dtype) for i in range(len(self._W_plastic))]

    def train(self, save_directory, save_prefix, logger, niters=100, verbose=False):
        if not self._init:
            raise Exception('cannot start training unless initialized')

        logger.error('rank %i will process %i neurons', self.rank, len(self._responsible_nids))
        if self.rank == 0:
            save_model_data(save_directory, save_prefix, self)

        if self._curr_iter == 0 and verbose and self.rank == 0:
            logger.error('intitial training error; %0.3f', self._training_error_history[-1])


        for citer in range(niters):
            if self.rank == 0:
                tic = clock.time() 

            for (t, time) in enumerate(list(self._times)):                 
                r, errors = None, None
                if self.rank == 0:
                    if t == 0:
                        x = deepcopy(self._y0)
                    dxdt = chaotic_model(t, x, self._J, self._H[:,t], self._tau)
                    x  += self._dt * dxdt
                    if update:
                        r  = np.tanh(x).reshape(self._N,1)
                        yt = np.matmul(self._J, r).reshape(self._N,)
                        errors = (yt - self._Ytruth[:,t]).astype(self._float_dtype)
   
                if update:
                    self.r = self.comm.bcast(r, root=0)
                    self.e = self.comm.bcast(errors, root=0)
                    delJ = self._update_model_parameters()
                    

                    # Update if at least one rank has more than one neuron
                    if self.rank == 0:
                        recdelJ = np.empty(sum(self.sendcounts), dtype=self._float_dtype)
                    else:
                        recdelJ = None

                    self.comm.Gatherv(delJ, (recdelJ, self.sendcounts), root=0)
                    if self.rank == 0:
                        cloc = 0
                        for i in range(len(self.all_nids)):
                            cnids   = self.all_nids[i]
                            num_ids = len(cnids)
                            leap    = num_ids * self._N
                            cdJ     = recdelJ[cloc:cloc + leap].reshape(num_ids, self._N)
                            self._J[cnids,:] -= (np.multiply(cdJ, self._S[cnids,:]))
                            cloc += leap
                            
                    # update if one neuron per rank
                    #recdelJ = None
                    #if self.rank == 0:
                    #    recdelJ = np.empty((self.size, self._N), dtype=self._float_dtype)
                    #self.comm.Gather(delJ, recdelJ, root=0)
                    #if self.rank == 0:
                    #    for i in range(len(self.all_nids)):
                    #        cnids = self.all_nids[i]
                    #        self._J[cnids,:] -= np.multiply(recdelJ[i], self._S[cnids,:])
                    
            if self.rank == 0:
                self._evaluate_model()
                self._curr_iter += 1
                if verbose:
                    elapsed = clock.time() - tic
                    logger.error('iteration %i took %0.3f seconds', self._curr_iter, elapsed)
                    logger.error('iteration training error: %f', self._training_error_history[-1])
                    logger.error('---')
                save_model_data(save_directory, save_prefix, self)
 
    def _update_model_parameters(self):       
        delJ = []
        for (i, nid) in enumerate(self._responsible_nids):
            pids      = self._W_plastic[i] 
            r_plastic = self.r[pids] 
            P     = self._Pt[i]
            #PxR   = sgemm(1.0, P, r_plastic)
            #RxPxR = (1. + sgemv(1.0, r_plastic.T, PxR))[0]
            #Pupdate = (sgemm(1.0, PxR, PxR.T) / RxPxR)
            PxR     = np.dot(P, r_plastic)
            RxPxR   = 1. + np.matmul(r_plastic.T, PxR)[0][0]
            Pupdate = np.matmul(PxR, PxR.T) / RxPxR
            self._Pt[i] -= Pupdate.astype(self._float_dtype)
            cdelJ = (1./RxPxR) * self.e[nid] * PxR.reshape(-1,)

            cdelJvec       = np.zeros((self._N,)).astype(self._float_dtype)
            cdelJvec[pids] = cdelJ
            delJ.extend(cdelJvec)
        return np.asarray(delJ, dtype=self._float_dtype)

    def _evaluate_model(self):
            
        evaluate_kwargs = {'J': self._J, 'H': self._H, 'tau': self._tau, 'Y': self._Ytruth, 'dt': self._dt}
        model_current   = solve_stochastic_euler(self._times, self._y0, **evaluate_kwargs)
        activations     = np.tanh(model_current)
        #Ymodel      = sgemm(1.0, self._J, activations)
        Ymodel      = np.matmul(self._J, activations)
        
        #self._validation_error_history.append(self._calculate_validation_error(Ymodel))
        self._training_error_history.append(self._calculate_training_error(Ymodel))        
         
    def _calculate_validation_error(self, Ymodel):
        if self._validation_idxs is None:
            return -1.
        errors = (Ymodel - self._Ytruth) ** 2
        return np.sum(errors[:,self._validation_idxs] ** 2) / float(len(self._validation_times))
        
        
    def _calculate_training_error(self, Ymodel):
        sq_errors = (Ymodel - self._Ytruth) ** 2
        loss = np.sum(np.mean(sq_errors, axis=0))
        #loss = np.sum([np.mean(sq_errors[:,i]) for i in self._training_idxs])
        return loss / float(len(self._times))
            

 
def solve_stochastic_euler(times, y0, **kwargs):
    J, H, tau, dt = kwargs['J'], kwargs['H'], kwargs['tau'], kwargs['dt']
    iteration_inputs = []
    for (t,time) in enumerate(times):
        h = H[:,t]
        if t == 0:
            dydt         = chaotic_model(None, y0, J, h, tau)
            new_input = y0 + dt * dydt
        else:
            dydt         = chaotic_model(None, iteration_inputs[-1], J, h, tau)
            new_input    = iteration_inputs[-1] + dt * dydt
        iteration_inputs.append(deepcopy(new_input))
        
    iteration_inputs = np.asarray(iteration_inputs, dtype='float32')
    return iteration_inputs.T


def solve_bdf(times, y0, J, h, a, g):
    from scipy.integrate import solve_ivp
    f = lambda t, y: chaotic_model(t,y,J, h, u, a, g)
    sol = solve_ivp(f, [times[0], times[-1]], y0, t_eval=times)
    return sol.y.astype('float32')
    
def chaotic_model(t, y, J, h, tau):   
    
    tau_inv = np.float32(1./tau)
    y = y.astype('float32')
    h = h.astype('float32')
    #struct_scale = J1 / np.sqrt(float(nnodes))
    #row_balanced = (adj_matrix - (np.matmul(adj_matrix, np.outer(e,e)) / float(nnodes)))
    #dydt = (1./tau) * (-y + np.matmul(row_balanced, np.tanh(y)) + struct_scale * np.matmul(np.outer(e,v), np.tanh(y)))
    
    #Jty = sgemm(1.0, J, np.tanh(y)).reshape(-1,)
    tany = np.tanh(y)
    Jty  = np.matmul(J, tany)
    dydt = tau_inv * (-y + Jty + h)
    return dydt

def firing_rate_spatial_overlap(currents, N, e):
    overlap = 1./N * np.matmul(e.T, np.tanh(currents))
    return overlap.reshape(overlap.shape[1],)

def current_spatial_overlap(currents, N, e):
    overlap = 1./N * np.matmul(e.T, currents)
    return overlap.reshape(overlap.shape[1], )

def spatial_coherence(current_spatial_overlap, currents, N):
    return np.sqrt( np.mean(current_spatial_overlap ** 2) / ( (1./N) * np.sum(np.mean(currents**2,axis=1))) )
    


# Main network class as presented in:
#
#   Laje, R. and Buonomano, D.V. (2013). Robust timing and motor patterns by taming chaos in recurrent neural networks. Nat Neurosci.
#
# Author: Julien Vitay (julien.vitay@informatik.tu-chemnitz.de)
# Licence: MIT

class RecurrentNetwork(object):
    """
    Class implementing a recurrent network with read-out weights and RLS learning rules.
    **Parameters:**
    * Ni : Number of input neurons
    * N : Number of recurrent neurons
    * No : Number of read-out neurons
    * tau : Time constant of the neurons
    * g : Synaptic strength scaling
    * pc : Connection probability
    * Io : Noise variance
    * delta : Initial value of the P matrix
    * P_plastic : Percentage of neurons receiving plastic synapses
    * dtype: floating point precision (default: np.float32)
    """
    def __init__(self, N=800, Ni=2, tau=1.0, g=1.5, pc=0.1, Io=0.001, delta=1.0, P_plastic=0.6, dtype=np.float32):
        # Copy the parameters
        self.N = N
        self.Ni = Ni
        self.tau = tau
        self.g = g
        self.pc = pc
        self.Io = Io
        self.delta = delta
        self.P_plastic = P_plastic
        self.N_plastic = int(self.P_plastic*self.N) # Number of plastic cells = 480
        self.dtype = dtype

        # Build the network
        self.build()

    def build(self):
        """
        Initializes the network including the weight matrices.
        """

        # Recurrent population
        self.x = np.random.uniform(-1.0, 1.0, (self.N, 1)).astype(self.dtype)
        self.r = np.tanh(self.x).astype(self.dtype)

        # Read-out population
        self.z = np.zeros((self.N, 1), dtype=self.dtype)

        self.W_in = np.random.randn(self.N, self.Ni).astype(self.dtype)
        # Weights between the recurrent units
        self.W_rec = (np.random.randn(self.N, self.N) * self.g/np.sqrt(self.pc*self.N)).astype(self.dtype)

        # The connection pattern is sparse with p=0.1
        connectivity_mask = np.random.binomial(1, self.pc, (self.N, self.N))
        connectivity_mask[np.diag_indices(self.N)] = 0
        self.W_rec *= connectivity_mask

        # Store the pre-synaptic neurons to each plastic neuron
        self.W_plastic = [list(np.nonzero(connectivity_mask[i, :])[0]) for i in range(self.N_plastic)]

        # Inverse correlation matrix of inputs for learning recurrent weights
        self.P = [1./self.delta*np.identity(len(self.W_plastic[i])).astype(self.dtype) for i in range(self.N_plastic)]

    def simulate(self, times, stimulus=None, trajectory=np.array([]), noise=True, training=True, verbose=True):
        """
        Simulates the recurrent network for the given duration, with or without plasticity.
        * `stimulus`: np.array for the inputs. Determines the duration.
        * `noise`: if noise should be added to the recurrent units (default: True)
        * `trajectory`: during learning, defines which target function should be learned (default: no learning)
        * `learn_start`: time when learning should start.
        * `learn_stop`: time when learning should stop.
        * `learn_readout`: defines whether the recurrent (False) or readout (True) weights should be learned.
        * `verbose`: defines if the loss should be printed (default: True)
        """

        # Get the stimulus shape to know the duration

        # Arrays for recording
        record_r = np.zeros((len(times), self.N, 1), dtype=self.dtype)
        record_z = np.zeros((len(times), self.N, 1), dtype=self.dtype)
        self.P = [1./self.delta*np.identity(len(self.W_plastic[i])).astype(self.dtype) for i in range(self.N_plastic)]

        # Reset the recurrent population
        self.x = np.random.uniform(-1.0, 1.0, (self.N, 1)).astype(self.dtype)
        self.r = np.tanh(self.x).astype(self.dtype)

        # Reset loss term
        self.loss = 0.0

        # Ensure the floating point precision
        if stimulus is not None:
            stimulus = stimulus.astype(self.dtype)
        trajectory = trajectory.astype(self.dtype)

        # Simulate for the desired duration
        
        tic = clock.time()
        no_stim = False
        if stimulus is None:
            no_stim = True
        
        for t in range(len(times)):
            if no_stim:
                stimulus = np.random.normal(0, 0.05, size=(self.Ni,1))

            # Update the neurons' firing rates
            self.update_neurons(stimulus, noise)

            # Recording
            record_r[t, :, :] = self.r
            record_z[t, :, :] = self.z

            # Learning
            if training and t % 2 == 0:
                self.rls_recurrent(trajectory[:, t, :])


        # Print the loss at the end of the trial
        if trajectory.size > 0 and verbose:
            elapsed = clock.time() - tic
            print('Loss:', self.loss/float(len(times)))
            print('that took %0.3f seconds' % elapsed)
        return record_r, record_z

    def update_neurons(self, stimulus, noise):
        """
        Updates neural variables for a single simulation step.
        """
        # Inputs are set externally
        # Noise can be shut off
        self.I = stimulus
        I_noise = (self.Io * np.random.randn(self.N, 1) if noise else np.zeros((self.N, 1))).astype(self.dtype)
        # tau * dx/dt + x = I + sum(r) + I_noise
        self.x += 0.25 * (np.dot(self.W_in, self.I) + np.dot(self.W_rec, self.r) + I_noise - self.x)/self.tau
        # r = tanh(x)
        self.r = np.tanh(self.x)
        # z = sum(r)
        self.z = np.dot(self.W_rec, self.r)

    def rls_recurrent(self, target):
        """
        Applies the RLS learning rule to the recurrent weights.
        """
        # Compute the error of the recurrent neurons
        error = self.z - target
        self.loss += np.mean(error**2)

        # Apply the FORCE learning rule to the recurrent weights
        for i in range(self.N_plastic): # for each plastic post neuron
            # Get the rates from the plastic synapses only
            r_plastic = self.r[self.W_plastic[i]]
            # Multiply with the inverse correlation matrix P*R
            PxR = np.dot(self.P[i], r_plastic)
            # Normalization term 1 + R'*P*R
            RxPxR = (1. + np.dot(r_plastic.T,  PxR))
            # Update the inverse correlation matrix P <- P - ((P*R)*(P*R)')/(1+R'*P*R)
            self.P[i] -= np.dot(PxR, PxR.T)/RxPxR
            # Learning rule W <- W - e * (P*R)/(1+R'*P*R)
            self.W_rec[i, self.W_plastic[i]] -= error[i, 0] * (PxR/RxPxR)[:, 0]

 
