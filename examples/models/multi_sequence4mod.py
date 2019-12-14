from __future__ import division

import numpy as np

from pycog import tasktools

#-----------------------------------------------------------------------------------------
# Network structure
#-----------------------------------------------------------------------------------------

Nin = 1
N = 200
TypeCount = 4
NoutSplit = 5
Nout = TypeCount * NoutSplit

# E/I
ei, EXC, INH = tasktools.generate_ei(N)

# Time constant
tau = 50

gamma_k = 5

#-----------------------------------------------------------------------------------------
# Noise
#-----------------------------------------------------------------------------------------

var_rec = 0.01**2
var_in = np.array([0.003**2])

def mode_const(x):
    return 0

def mode_sin(x):
    return np.sin(x * (np.pi * 2))

def mode_square(x):
    if x - int(x) > 0.5:
        return -1
    else:
        return 1

def mode_tri(x):
    frac = x - int(x)
    if frac > 0.8:
        return -1
    elif frac > 0.3 and frac < 0.5:
        return 1
    else:
        return 0

modes = [mode_const, mode_sin, mode_square, mode_tri]

def generate_trial(rng, dt, params):
    T = 1000

    signal_time = rng.uniform(100, T - 800)

    output_delay = 100

    width = 100
    magnitude = 4

    epochs = {}
    epochs['T'] = T
    t, e  = tasktools.get_epochs_idx(dt, epochs) # Time, task epochs in discrete time
    trial = {'t': t, 'epochs': epochs}           # Trial

    trial['info'] = {}

    signal_time /= dt
    width /= dt
    output_delay /= dt

    input_type = 3#rng.randint(0, TypeCount)

    # Input matrix
    X = np.zeros((len(t), Nin))

    for tt in range(len(t)):
        if tt >= signal_time:
            X[tt][0] = (input_type + 1) * 2 * np.exp(-(tt - signal_time) / (output_delay * 4))

    #X = np.flip(X)
    trial['inputs'] = X

    #---------------------------------------------------------------------------------
    # Target output
    #---------------------------------------------------------------------------------

    if params.get('target_output', False):
        Y = np.zeros((len(t), Nout)) # Output matrix
        M = np.ones((len(t), Nout)) # Mask matrix

        for i in range(NoutSplit):
            for tt in range(len(t)):
                Y[tt][i + NoutSplit * input_type] = \
                    np.exp( -(tt - (signal_time + output_delay * (i + 2))) \
                    **2 / (2 * width**2)) * magnitude

        trial['outputs'] = Y

    return trial

min_error = 0.1

n_validation = 100


if __name__ == '__main__':
    from pycog          import RNN
    from pycog.figtools import Figure

    rnn  = RNN('work/data/multi_sequence4mod/multi_sequence4mod.pkl', {'dt': 0.5, 'var_rec': 0.01**2,
        'var_in':  np.array([0.003**2])})
    trial_args = {}
    info = rnn.run(inputs=(generate_trial, trial_args), seed=72)

    fig  = Figure()
    plot = fig.add()

    colors = ['red', 'green', 'yellow', 'orange', 'purple', 'cyan', 'magenta', 'pink']

    plot.plot(rnn.t/tau, rnn.u[0], color=Figure.colors('blue'), label='$input$')
    for i in range(Nout):
        if i % NoutSplit == 0:
            k = {'label': '$output%d$'%int(i / NoutSplit)}
        else:
            k = {}
        plot.plot(rnn.t/tau, rnn.z[i], color=Figure.colors(colors[int(i / NoutSplit)]), **k)

    Nexc = int(N * 0.8)
    np.savetxt("rnnt.txt", rnn.t/tau)
    np.savetxt("r4.txt", np.divide(rnn.r[:Nexc].mean(axis=0), rnn.r[Nexc:].mean(axis=0)))

    plot.xlim(rnn.t[0]/tau, rnn.t[-1]/tau)
    plot.ylim(0, 15)

    prop = {'prop': {'size': 20}, 'handlelength': 1.2,
        'handletextpad': 1.1, 'labelspacing': 0.5}
    plot.legend(bbox_to_anchor=(0.81, 1), **prop)

    plot.xlabel(r'$t/\tau$')
    plot.ylabel(r'$input \& output$')

    fig.save(path='.', name='multi_sequence4mod')

    print(rnn.Wout)


    fig  = Figure()
    plot = fig.add()
    plot.hist(np.asarray(rnn.Wrec).reshape(-1), bins = 100)
    plot.xlabel(r'$W_{rec}\mbox{ } weight$')
    plot.ylabel(r'$Distribution$')
    fig.save(path='.', name='multi_sequence4mod-weight')
