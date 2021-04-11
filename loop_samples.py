# Version: 0.2, 2021-04-11
# Author: Jonas Tallhage, jonas@tallhage.se, https://github.com/jonastallhage
# License: MIT
# Documentation: https://github.com/jonastallhage/sampling-tools
#                Use python loop_samples.py -h to see input parameters


import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
import os
import sys
import argparse
from smptoollib import wavtools
import itertools
import time
import multiprocessing as mp
import json
import threading
import colorama
import datetime
from types import SimpleNamespace


def get_zero_crossings(waveform, crossing_type = 'positive'):
    waveform_no_zeros = waveform + 0.1 * np.ones(waveform.shape)
    if crossing_type == 'negative':
        crossings = np.where(np.diff(np.sign(waveform_no_zeros)) < 0)
    else:
        crossings = np.where(np.diff(np.sign(waveform_no_zeros)) > 0)
    return crossings[0]


def get_cycle_indices(waveform, crossing_type = 'positive'):
    waveform_no_zeros = waveform + 0.1 * np.ones(waveform.shape)
    if crossing_type == 'negative':
        crossings = np.where(np.diff(np.sign(waveform_no_zeros)) < 0)
    else:
        crossings = np.where(np.diff(np.sign(waveform_no_zeros)) > 0)
    cycle_indices = []
    for k in range(len(crossings[0])-1):
        cycle_indices.append((crossings[0][k],crossings[0][k+1]))
    return cycle_indices


def print_cycle_indices(cycle_indices):
    for idx in cycle_indices:
        print(idx)


def plot_cycles(waveform, indices, indices_to_plot = []):
    plt.figure('Cycles')
    if len(indices_to_plot):
        for idx_n in indices_to_plot:
            idx = indices[idx_n]
            plt.plot(waveform[idx[0]:idx[1]])
    else:
        for idx in indices:
            plt.plot(waveform[idx[0]:idx[1]])


def plot_from_index_n_to_m(waveform, indices, index_n, index_m):
    plt.figure('From n to m')
    plt.plot(waveform[indices[index_n][0]:indices[index_m][1]])


def get_cycles(waveform, crossing_type = 'positive'):
    if crossing_type == 'negative':
        crossings = np.where(np.diff(np.sign(waveform)) < 0)
    else:
        crossings = np.where(np.diff(np.sign(waveform)) > 0)
    cycles = []
    for k in range(len(crossings[0])-1):
        cycles.append(waveform[crossings[0][k]:crossings[0][k+1]])
    return cycles


def compare_indices(waveform, indices, reference_idx_nbr, compare_idx_nbr_range):
    # Returns a list of tuples on the form (compare_idx_nbr, distance), the list
    # is sorted with the smallest distance first. I.e. the first element contains
    # the index number of the cycle which is closest to the reference cycle. We
    # should pick the start of this cycle as the end sample of our loop.
    ref_idx = indices[reference_idx_nbr]
    ref_cycle = waveform[ref_idx[0]:ref_idx[1]]
    ref_len = len(ref_cycle)
    distances = []
    for idxnbr in range(compare_idx_nbr_range[0], compare_idx_nbr_range[1]):
        idx = indices[idxnbr]
        cycle = waveform[idx[0]:idx[1]]
        if len(cycle) > ref_len:
            dif = cycle[0:ref_len] - ref_cycle
        elif len(cycle) < ref_len:
            dif = cycle - ref_cycle[0:len(cycle)]
        else:
            dif = cycle - ref_cycle
        distances.append((idxnbr, sum(abs(dif))))
    # Sort the distances list by least distance, first element is the index nbr
    distances = sorted(distances, key=lambda x: x[1])
    return distances


def get_loop_idx(start_cycle_idx, best_fit_cycle_idx):
    loop_start_sample = start_cycle_idx[0]
    loop_end_sample = best_fit_cycle_idx[0]
    return (loop_start_sample, loop_end_sample)


def get_average_samples_per_cycle(waveform, crossing_type):
    # May want to add a get_samples_per_cycle function
    idxs = get_cycle_indices(waveform, crossing_type)
    lengths = [(idx[1]-idx[0]) for idx in idxs]
    avg_len = sum(lengths)/len(lengths)
    return int(avg_len)


def get_audition_clip(waveform, loop_idx, t_pre, t_post, fs):
    # Indices of the pre- and post-loop-point clips
    pre_idx = (loop_idx[1]-int(t_pre*fs), loop_idx[1])
    post_idx = (loop_idx[0], loop_idx[0]+int(t_post*fs))
    return np.concatenate((waveform[pre_idx[0]:pre_idx[1]], waveform[post_idx[0]:post_idx[1]]))


def write_audition_file(waveform, fs, loop_idx, audition_length, output_dir, filename):
    t_pre = audition_length/2
    t_post = audition_length/2
    audition_clip = get_audition_clip(waveform, loop_idx, t_pre, t_post, fs)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    audition_file = output_dir + "/" + filename
    wavfile.write(audition_file, fs, audition_clip)


def save_loop_point_plot(waveform, samples_per_cycle, loop_idx, plot_length,
                         diagnostics_dir, filename, plot_counter, n_plots):
    plt.ioff()
    pre_post = plot_length*samples_per_cycle/2
    plot_clip = get_audition_clip(waveform, loop_idx, pre_post, pre_post, 1)
    plot_file = diagnostics_dir + "/" + filename.split('.')[0] + ".jpg"
    plt.figure(filename)
    plt.title(f'{plot_counter}/{n_plots}: {filename}')
    plt.plot(plot_clip)
    axes = plt.gca()
    axes.axvline(x=pre_post, ymin=0, ymax=1,
                 linestyle='--', color='gray', linewidth=0.5, zorder = 0)
    plt.savefig(plot_file)
    plt.close()
    plt.ion()


def get_batches(start_candidates, N_workers):
    candidates = list(start_candidates)
    batches = []
    for k in range(N_workers):
        batches.append(candidates[k:-1:N_workers])
    # This might be avoidable by using candidates[k::N_workers] instead in the line above
    batches[-1].append(candidates[-1])
    return batches
    

def process_start_candidates(start_candidates, N_start_candidates, crossings, minlength, cmplen, weights, w, mindist, q):
    n = 1
    bestloop = (None, np.inf)
    for sc in start_candidates:
        end_candidates = crossings[np.where(crossings - sc >= minlength)]
        for ec in end_candidates:
            # Since we want a midpoint with unity weight we have made the weights
            # one step longer than the pre- and post-windows, we add 1 to the
            # end-of-slice index to make the slice the same length as the window.
            # The end must be chosen to get the unity point to be at the start
            # candidate index.
            # Note that at present a rectangular window is used, this will
            # only be important if other windowing techniques are added in a
            # future update since all weights in the rectangular window
            # are unity.
            ss = (int(sc-cmplen[0]),int(sc+cmplen[1]+1)) # Start slice
            es = (int(ec-cmplen[0]),int(ec+cmplen[1]+1)) # End slice
            
            # This can be uncommented to test that the unity point is at the start candidate
            # wss = weights*w[ss[0]:ss[1]]
            # wes = weights*w[es[0]:es[1]]
            # print(f'sdaf: {w[sc]}:{wss[cmplen[0]+0]} ; {w[ec]}:{wes[cmplen[1]+0]}')

            distance = sum(weights*abs(w[ss[0]:ss[1]] - w[es[0]:es[1]]))
            mindist = distance if (distance < mindist) else mindist
            bestloop = ((sc,ec),distance) if (distance < bestloop[1]) else bestloop
        q.put({'event_type': 'candidate_done'})
        n += 1
    return bestloop


def status_task(q, N_files, config, aux):
    # Run in a separate thread to write status to terminal and values to CSV file during processing,
    # queue object q is used for inputs.
    n_done = 0
    N_total = 0
    while True:
        try:
            msg = q.get()
            if msg['event_type'] == 'start_processing':
                n_done = 0
                N_total = msg['N_start_candidates']
                filenbr = msg['filenbr']
                filename = msg['filename']
                print(f"{filenbr+1}/{N_files}: {filename}")
            elif msg['event_type'] == 'candidate_done':
                n_done += 1
                print(f"\r   Start candidate {n_done} / {N_total}", end='', flush=True)
            elif msg['event_type'] == 'loop_found':
                process_time = msg['process_time']
                filename = msg['filename']
                bestloop = msg['bestloop']
                print(f"\n   Multi-process took {process_time} s")
                if config.write_csv:
                    write_to_csv(filename, bestloop, config, aux)
        except:
            pass

def get_weights(weighting, winlen):
    if weighting == 'ones':
        precrossing = np.ones(winlen[0])
        postcrossing = np.ones(winlen[1])
        midpoint = np.ones(1)
        weights = np.concatenate((precrossing, midpoint, postcrossing))
    return weights
    


def get_loop_using_crossings_mp(w, fs, filename, filenbr, config, pool, q, N_workers):
    crossing_type = config.crossing_type
    loop_region_start = config.loop_start
    loop_region_end = config.loop_end
    min_loop_length = config.min_loop_length
    compare_length = config.compare_length
    weighting = config.weighting
    
    winlen = (int(compare_length[0]*fs), int(compare_length[1]*fs)) # Compare slice length
    
    # This may return crossing indices which are too close to the start/end which
    # would give us errors when trying to access a slice due to the windowing,
    # hence we need to restrict the crossings to ones which cannot produce this effect
    crossings_all = get_zero_crossings(w, crossing_type)
    crossings_all = crossings_all[np.where(crossings_all > winlen[0])]
    crossings_all = crossings_all[np.where(crossings_all < (len(w) - winlen[1]))]
    crossings = crossings_all[np.where((crossings_all >= loop_region_start*fs) &
                                       (crossings_all <= loop_region_end*fs))]
    
    minlength = min_loop_length*fs
    weights = get_weights(weighting, winlen)
    mindist = np.inf
    
    # To speed things up we want to break all the start candidates up into batches which can be
    # run in separate processes.
    start_candidates = crossings[np.where(crossings[-1] - crossings >= minlength)]
    batches = get_batches(start_candidates, N_workers)
    N_start_candidates = len(start_candidates)
    
    q.put({'event_type': 'start_processing',
            'N_start_candidates': N_start_candidates,
            'filename': filename,
            'filenbr': filenbr})
    
    resultlist = []

    # Distribute the batches across a number (set by N_workers) of processes so they can be run in parallel.
    for k in range(N_workers):
        result = pool.apply_async(process_start_candidates,
                                  (batches[k], N_start_candidates, crossings, minlength, winlen, weights, w, mindist, q))
        resultlist.append(result)
    
    looplist = []
    for result_nbr,result in enumerate(resultlist):
        try:
            looplist.append(result.get())
        except:
            print(f"Exception when trying to get result {result_nbr}")
    
    bestloop = (None, np.inf)
    for loop in looplist:
        bestloop = loop if (loop[1] < bestloop[1]) else bestloop
        
    # Normalize distance to nbr of samples used in the compare length
    looplength_seconds = (bestloop[0][1]-bestloop[0][0])/fs
    bestloop = (bestloop[0],bestloop[1]/(winlen[0]+winlen[1]),looplength_seconds)
    return bestloop, N_start_candidates
  

def get_wavfile_data(filename):
    f = open(filename, 'rb')
    wav = wavtools.WavInterface(f)
    f.close()
    w = wav.data
    fs = wav.fs
    return w, fs    


def write_diagnostics(w, fs, loop_idx, filename, filenbr, N_files, config, aux):
    samples_per_cycle = get_average_samples_per_cycle(w, config.crossing_type)    
    if not os.path.exists(aux.diagnostics_dir):
        os.makedirs(aux.diagnostics_dir)
    filename_base = os.path.split(filename)[1]
    if config.audition_length:
        write_audition_file(w, fs, loop_idx, config.audition_length, aux.diagnostics_dir, filename_base)
    if config.plot_length:
        plot_counter = filenbr + 1
        save_loop_point_plot(w, samples_per_cycle, loop_idx, config.plot_length,
                             aux.diagnostics_dir, filename_base, plot_counter, N_files)

    
def write_output_wav(w, fs, loop_idx, filename, config, aux):
    filename_base = os.path.split(filename)[1]
    outdir = aux.looped_samples_dir
    if not os.path.exists(outdir):
        os.makedirs(outdir)   
    output_filename = outdir + "/" + filename_base
    if config.method == 'truncate':
        wavfile.write(output_filename, fs, w[loop_idx[0]:loop_idx[1]])
    else:
        # TODO: This could be made more efficient by passing around a reference to the previously instantiated WavInterface
        f = open(filename, 'rb')
        wav = wavtools.WavInterface(f)
        wav.write_with_loop(output_filename, int(loop_idx[0]), int(loop_idx[1]))
        f.close()
    
    
def process_file(filename, filenbr, N_files, config, pool, q, N_workers, aux):
    w, fs = get_wavfile_data(filename)
    
    t_start = time.perf_counter()
    bestloop, N_start_candidates = get_loop_using_crossings_mp(w, fs, filename, filenbr, config, pool, q, N_workers)
    t_stop = time.perf_counter()

    loop_idx = bestloop[0]
    
    q.put({'event_type': 'loop_found',
           'process_time': t_stop - t_start,
           'filenbr': filenbr,
           'filename': filename,
           'bestloop': bestloop,
           'looplength': (bestloop[0][1]-bestloop[0][0])/fs,
           'N_start_candidates': N_start_candidates})
    
    if config.write_diagnostics:
        write_diagnostics(w, fs, loop_idx, filename, filenbr, N_files, config, aux)
    if config.write_looped_samples:
        write_output_wav(w, fs, loop_idx, filename, config, aux)   

        
def write_to_csv(wav_filename, bestloop, config, aux):
    wav_filename_base = os.path.split(wav_filename)[1]
    loop_idx = bestloop[0]
    loop_distance = bestloop[1]
    loop_length = bestloop[2]
    output_str = f"{wav_filename_base}, {loop_idx[0]}, {loop_idx[1]}, {loop_distance}, {loop_length}\n"
    with open(aux.csv_filename, 'a') as f:
        f.write(output_str)


def write_csv_header(config, aux):
    with open(aux.csv_filename, 'w') as f:
            f.write('--- CONFIG ---\n')
            for key in config.__dict__.keys():
                f.write(f"{key}, {config.__dict__[key]}\n")
            f.write('\n--- LOOP POINTS FORMAT ---\n')
            f.write('filename, loop_start_sample, loop_end_sample, loop_distance, loop_length_seconds\n')
            f.write('--- Notes: End sample is exclusive to conform to Python indexing. If using to ---\n')
            f.write('---        embed loop points in .wav, set end sample to (<loop end sample> - 1).  ---\n')
            f.write('---        loop_distance is normalized to compare length and thus comparable ---\n')
            f.write('---        between settings, it is not normalized to overall levels and thus not ---\n')
            f.write('---        comparable between files with different levels  ---\n')
            f.write('\n--- LOOP POINTS ---\n')


def get_parser_args():
    # TODO: Use a more pythonic way of defining input arguments, e.g. a list of dicts with loops to populate
    #       the argument and json parsers, this should make it easier to allow argparser input to override file
    #       input. Possibly shift to yaml for file input.
    parser = argparse.ArgumentParser()
    parser.add_argument('-cfg', '--config', type=str,
                        help='E.g. myconfigfile.json, see included file for available fields, will override all other options',
                        default=None)
    parser.add_argument('-i', '--indir', type=str,
                        help='Directory containing input samples, all .wav files will be examined for loop points',
                        default='.')
    parser.add_argument('-c', '--crossing-type', type=str,
                        help='positive or negative',
                        default='negative')
    parser.add_argument('-s', '--loop-start', type=float,
                        help='Approximate start of the loop, in seconds',
                        default=0)
    # TODO: Use <= 0 values to indicate that approximate end should be at the end of the input file
    parser.add_argument('-e', '--loop-end', type=float,
                        help='Approximate end of the loop, in seconds',
                        default=1)
    parser.add_argument('-L', '--min-loop-length', type=float,
                        help='Minimum loop length (in seconds), must be smaller than loop-end minus loop-start',
                        default=-1)
    parser.add_argument('-m', '--method', type=str,
                        help='loop-points or truncate',
                        default='truncate') 
    parser.add_argument('-a', '--audition-length', type=float,
                        help='Length (in seconds) of loop audition clips, these will be saved to diagnose subfolder with the loop point in the middle of the clip',
                        default=3)
    parser.add_argument('-p', '--plot-length', type=float,
                        help='Length (in cycles) of diagnostic plots showing the waveform immediately around the loop point',
                        default=3)
    parser.add_argument('-d', '--write-diagnostics', type=int,
                        help='Write diagnostics (1 for true, 0 for false)',
                        default=1)
    parser.add_argument('-csv', '--write-csv', type=int,
                        help='Write csv file with loop points, distance etc. (1 for true, 0 for false)',
                        default=1)
    parser.add_argument('-cpr', '--compare-length-pre', type=float,
                        help='Length (in seconds) of pre-loop-point compare window',
                        default=0.01) 
    parser.add_argument('-cpo', '--compare-length-post', type=float,
                        help='Length (in seconds) of pre-loop-point compare window',
                        default=0.01) 
    parser.add_argument('-w', '--weighting', type=str,
                        help='Type of weights for comparison (valid options: ones)',
                        default='ones')
    parser.add_argument('-wr', '--write-looped-samples', type=str,
                        help='Write looped samples',
                        default=1)
    parser.add_argument('-N', '--N-workers', type=int,
                        help='Number of worker processes, for fastest processing set this equal to number of CPU cores',
                        default=7)
    args = parser.parse_args()
    if args.config:
        with open(args.config) as f:
            config_str = f.read()
        config = json.loads(config_str)
        args.indir = config['Input dir']
        args.crossing_type = config['Crossing type']
        args.loop_start = config['Loop start']
        args.loop_end = config['Loop end']
        args.min_loop_length = config['Min loop length']
        args.method = config['Method']
        args.audition_length = config['Audition length']
        args.plot_length = config['Plot length']       
        args.write_diagnostics = config['Write diagnostics']
        args.write_csv = config['Write CSV']
        args.compare_length_pre = config['Compare length pre']
        args.compare_length_post = config['Compare length post']
        args.weighting = config['Weighting']
        args.write_looped_samples = config['Write looped samples']
        args.N_workers = config['N workers']
    if (args.min_loop_length <= 0) or (args.min_loop_length > args.loop_end - args.loop_start):
        args.min_loop_length = args.loop_end - args.loop_start - 0.1
    args.compare_length = (args.compare_length_pre, args.compare_length_post)
    print("Input dir:", args.indir)
    print("Crossing type:", args.crossing_type)
    print("Loop start (approximate):", args.loop_start)
    print("Loop end (approximate):", args.loop_end)
    print("Min loop length:", args.min_loop_length)
    print("Method:", args.method)
    print("Audition length:", args.audition_length, "seconds")
    print("Plot length:", args.plot_length, "cycles")
    print("Write diagnostics:", args.write_diagnostics)
    print("Write CSV:", args.write_csv)
    print("Compare length:", args.compare_length)
    print("Weighting:", args.weighting)
    print("Write looped samples:", args.write_looped_samples)
    print("N workers:", args.N_workers)
    return args


if __name__ == "__main__":
    t_start = time.perf_counter()
    colorama.init()
            
    args = get_parser_args()
    config = args
    
    print("")
    
    plot_counter = 1
    N_files = 0
    for f_name in os.listdir(args.indir):
        if f_name.split('.')[-1] == 'wav':
            N_files += 1
    
    filenbr = 0
    
    aux = SimpleNamespace()
    
    dt = datetime.datetime.today()
    aux.dt_str = f"{dt.year}{dt.month:02}{dt.day:02}_{dt.hour:02}{dt.minute:02}{dt.second:02}"
    aux.outdir = config.indir + '/autoloop_output_' + aux.dt_str
    aux.diagnostics_dir = aux.outdir + '/diagnostics'
    aux.looped_samples_dir = aux.outdir + '/looped_samples'
    aux.csv_filename = aux.outdir + '/loop_points.csv'
    if config.write_csv or config.write_diagnostics:
        if not os.path.exists(aux.outdir):
            os.makedirs(aux.outdir)
    aux.csv_filename = aux.outdir + '/loop_points.csv'
    
    if config.write_csv:
        write_csv_header(config, aux)   
    
    N_workers = args.N_workers
    pool = mp.Pool(processes=N_workers)
    m = mp.Manager()
    q = m.Queue()
    
    status_thread = threading.Thread(target=status_task,
                                     args=(q, N_files, config, aux),
                                     daemon=True)
    status_thread.start()
    
    for f_name in os.listdir(args.indir):
        if not(f_name.split('.')[-1] == 'wav'):
            continue
        f_full = args.indir + "/" + f_name
        process_file(f_full, filenbr, N_files, config, pool, q, N_workers, aux)
        filenbr += 1
        
    t_stop = time.perf_counter()
    time.sleep(0.1)
    print("")
    print(f"Total time: {t_stop-t_start} s")
    
    