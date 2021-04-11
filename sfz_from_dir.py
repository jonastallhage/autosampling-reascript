# Version: 0.1, 2020-07-27
# Author: Jonas Tallhage, jonas@tallhage.se, https://github.com/jonastallhage
# License: MIT
# Documentation: https://github.com/jonastallhage/sampling-tools
#                Use python sfz_from_dir.py -h to see input parameters

from types import SimpleNamespace
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-s', '--sampledir', type=str, default='.')
parser.add_argument('-o', '--outname', type=str, default='./my_sfz.sfz')
parser.add_argument('-f', '--format', type=str,
                    help='E.g. ignore_rootkey_lowkey_highkey_lowvelocity_highvelocity_roundrobin_groupname',
                    default='ignore_rootkey_lowkey_highkey_lowvelocity_highvelocity_roundrobin')
parser.add_argument('-t', '--filetype', type=str, help='E.g. wav', default='wav')
parser.add_argument('-l', '--loopmode', type=str,
                    help='no_loop, one_shot, loop_continuous or loop_sustain; use loop_continuous for oscillator looping',
                    default='no_loop')

args = parser.parse_args()

print("Sample dir: ", args.sampledir)
print("Format: ", args.format)
print("Output name: ", args.outname)
print("File type: ", args.filetype)
print("Loop mode: ", args.loopmode)
print("")

format_list = args.format.split('_')
groups = {}

f = open(args.outname, 'w')

f.write('<control>\n')
f.write(f'default_path={args.sampledir}/ // Relative path of your samples\n')
f.write('\n\n')

f.write('<global>\n')
f.write('// Parameters that affect the whole instrument go here\n')
f.write(f'loop_mode={args.loopmode}\n')
f.write('\n\n')

roundrobin_offset = 0
seq_length = 0

for s_name in os.listdir(args.sampledir):
    if not(s_name.split('.')[-1] == args.filetype):
        continue
    name_fields = s_name.split('.')[0].split('_')
    sample_params = {key:value for (key,value) in zip(format_list, name_fields)}
    if sample_params.get('roundrobin') == '0':
        roundrobin_offset = 1
    if int(sample_params.get('roundrobin', '0')) + roundrobin_offset > seq_length:
        seq_length = int(sample_params.get('roundrobin', '0')) + roundrobin_offset
    sample_params['filename'] = s_name
    print("Parsing", s_name)
    if not(sample_params.get('groupname', 'default') in groups.keys()):
        groups[sample_params.get('groupname', 'default')] = []
    groups[sample_params.get('groupname', 'default')].append(sample_params)
            

for key in groups:
    if seq_length > 0:
        f.write(f'<group> seq_length={seq_length}\n')
    else:
        f.write(f'<group>\n')
    for element in groups[key]:
        region_str = f'<region> ' \
                     f'sample={element["filename"]} ' \
                     f'pitch_keycenter={element["rootkey"]} ' \
                     f'lokey={element["lowkey"]} ' \
                     f'hikey={element["highkey"]} ' \
                     f'lovel={element["lowvelocity"]} ' \
                     f'hivel={element["highvelocity"]}'
        if seq_length > 0:
            region_str += f' seq_position={int(element["roundrobin"])+roundrobin_offset}\n'
        else:
            region_str += '\n'
        f.write(region_str)
    f.write('\n\n')

f.close()