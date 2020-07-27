# Version: 0.1, 2020-07-27
# Author: Jonas Tallhage, jonas@tallhage.se, https://github.com/jonastallhage
# License: MIT
# Documentation: https://github.com/jonastallhage/sampling-tools
#                Use python loop_samples.py -h to see input parameters

import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
import os
import time
import argparse

def get_cycle_indices(waveform, crossing_type = 'positive'):
    waveform_no_zeros = waveform + 0.1 * np.ones(waveform.shape)
    if crossing_type == 'negative':
        crossings = np.where(np.diff(np.sign(waveform_no_zeros)) < 0)
    else:
        crossings = np.where(np.diff(np.sign(waveform_no_zeros)) > 0)
    cycle_indices = []
    for k in range(len(crossings[0])-1):
        pass
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
                         output_dir, filename, plot_counter):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    pre_post = plot_length*samples_per_cycle/2
    plot_clip = get_audition_clip(waveform, loop_idx, pre_post, pre_post, 1)
    plot_file = diagnostics_dir + "/" + filename.split('.')[0] + ".jpg"
    plt.figure(filename)
    plt.title(f'{plot_counter}: {filename}')
    plt.plot(plot_clip)
    axes = plt.gca()
    # axes.axhline(linestyle='--', color='gray', linewidth=0.5, zorder = 0)
    axes.axvline(x=pre_post, ymin=0, ymax=1,
                 linestyle='--', color='gray', linewidth=0.5, zorder = 0)
    plt.savefig(plot_file)
    plt.close()

def get_parser_args():   
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--indir', type=str, default='.')
    parser.add_argument('-o', '--outdir', type=str, default='./looped_samples')
    parser.add_argument('-a', '--audition-length', type=int,
                        help='Length of loop audition clips, these will be saved to diagnose subfolder with the loop point in the middle of the clip',
                        default=3)
    parser.add_argument('-p', '--plot-length', type=int,
                        help='Length (in cycles) of diagnostic plots showing the waveform immediately around the loop point',
                        default=3)
    parser.add_argument('-c', '--crossing-type', type=str,
                        help='positive or negative',
                        default='negative')
    parser.add_argument('-s', '--loop-start', type=int,
                        help='Approximate start of the loop, in seconds',
                        default=0)                    
    args = parser.parse_args()
    print("Input dir:", args.indir)
    print("Output dir:", args.outdir)
    print("Crossing type:", args.crossing_type)
    print("Loop start:", args.loop_start)
    print("Audition length:", args.audition_length, "seconds")
    print("Plot length:", args.plot_length, "cycles")
    if not os.path.exists(args.outdir):
        os.makedirs(args.outdir)
    return args

if __name__ == "__main__":
            
    args = get_parser_args()
    
    plt.ioff()
    
    plot_counter = 1
    
    for f_name in os.listdir(args.indir):   
        f_full = args.indir + "/" + f_name
        print("Processing:", f_name)
        
        fs, w = wavfile.read(f_full)
        
        indices = get_cycle_indices(w, crossing_type = args.crossing_type)
        n_indices = len(indices)
    
        start_cycle_idx_nbr = next(x for x in enumerate(indices) if x[1][0] > fs*args.loop_start)[0]
        start_cycle_idx = indices[start_cycle_idx_nbr]
        samples_per_cycle = start_cycle_idx[1] - start_cycle_idx[0]
        distances = compare_indices(w, indices, start_cycle_idx_nbr, (n_indices-100,n_indices))
        best_fit_idx_nbr = distances[0][0]
        best_fit_idx = indices[best_fit_idx_nbr]
        
        loop_idx = get_loop_idx(start_cycle_idx, best_fit_idx)
        
        diagnostics_dir = args.outdir + "/diagnostics"
        if args.audition_length:
            write_audition_file(w, fs, loop_idx, args.audition_length, diagnostics_dir, f_name)
        if args.plot_length:
            save_loop_point_plot(w, samples_per_cycle, loop_idx, args.plot_length,
                                 diagnostics_dir, f_name, plot_counter)
            plot_counter += 1
        
        f_full_out = args.outdir + "/" + f_name
        wavfile.write(f_full_out, fs, w[loop_idx[0]:loop_idx[1]])
    
    plt.ion()