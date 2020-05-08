from mpi4py import MPI
import numpy as np
import time
from functional_analysis import NoiseAnalysis

def get_best(save_directory, save_prefix, T):
    val_errors, train_errors = [], []
    for i in range(1, T):
        filepath = save_directory + '/' + save_prefix + '-%i.npz' % i
        try:
            f = np.load(filepath, allow_pickle=True)
            val_errors.append(f['validation_error'])
            train_errors.append(f['train_error'])
            f.close()
        except:
            train_errors.append(1e9)
    best_iter = np.argmin(train_errors) + 1
    
    filepath  = save_directory + '/' + save_prefix + '-%i.npz' % best_iter
    f = np.load(filepath, allow_pickle=True)
    J = f['J']
    f.close()
    return J

# args for user: date, fish_id, state, niters

comm    = MPI.COMM_WORLD
date    = '1172020'
FISH_ID = 2
state   = 'preptz'

model_directory = '/mnt/f/dhh-soltesz-lab/zfish-modeling-outputs/%s/f%i/%s/models' % (date, FISH_ID, state)
info_filepath   = '/mnt/f/dhh-soltesz-lab/zfish-modeling-outputs/%s/f%i/%s/additional-info-%s.npz' % (date, FISH_ID, state, date)
model_prefix    = 'full-%s' % state

J = get_best(model_directory, model_prefix, 809)
noise_stds = [0.001, 0.005, 0.01, 0.025, 0.05, 0.075, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0]
trials_per_noise = 1
unique_y0        = 1
y0_range         = 0.1
save_filepath    = 'noise-analysis-fish-%i-state-%s.pkl' % (FISH_ID, state)

noise_analysis = NoiseAnalysis(J, info_filepath, noise_stds, trials_per_noise, unique_y0, y0_range, comm)
noise_analysis.initiate()
noise_analysis.run(save_filepath)
