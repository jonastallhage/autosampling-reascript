import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
import sounddevice
import os
import time
from types import SimpleNamespace

#1 - 4	"RIFF"	Marks the file as a riff file. Characters are each 1 byte long.
#5 - 8	File size (integer)	Size of the overall file - 8 bytes, in bytes (32-bit integer). Typically, you'd fill this in after creation.
#9 -12	"WAVE"	File Type Header. For our purposes, it always equals "WAVE".
#13-16	"fmt "	Format chunk marker. Includes trailing null
#17-20	16	Length of format data as listed above
#21-22	1	Type of format (1 is PCM) - 2 byte integer
#23-24	2	Number of Channels - 2 byte integer
#25-28	44100	Sample Rate - 32 byte integer. Common values are 44100 (CD), 48000 (DAT). Sample Rate = Number of Samples per second, or Hertz.
#29-32	176400	(Sample Rate * BitsPerSample * Channels) / 8.
#33-34	4	(BitsPerSample * Channels) / 8.1 - 8 bit mono2 - 8 bit stereo/16 bit mono4 - 16 bit stereo
#35-36	16	Bits per sample
#37-40	"data"	"data" chunk header. Marks the beginning of the data section.
#41-44	File size (data)	Size of the data section.

# Maybe use this for conversion instead of directly using int.from_bytes
def b2i(b):
    return int.from_bytes(b, 'little')
    
class WavInterface():
    def __init__(self, file):
        self.file = file
        if self.file:
            header_ok = self.__read_header()
            if header_ok:
                self.__read_data()
                self.__read_footer()
            else:
                print('Errors during parsing of file header, no data read')
    
    def __read_header(self):
        self.file.seek(0)
        self.h = self.file.read(36)
        is_riff = self.h[0:4].decode()
        if not(is_riff == 'RIFF'):
            print("Not an RIFF file")
            return False
        # Note: The file size excludes the two previous header parts, this field
        #       is thus "missing" 8 bytes (and so are all fields which indicate
        #       chunk size)
        self.file_size = int.from_bytes(self.h[4:8], 'little')
        self.riff_type = self.h[8:12].decode()
        if not(self.riff_type == 'WAVE'):
            print("RIFF type not WAVE (i.e. not a .wav file)")
            return False
        fmt_start = self.h[12:16].decode() # Indicates start of format fields
        if not(fmt_start == 'fmt '):
            print('Expected indication of fmt chunk start in bytes 13 to 16')
            return False
        # Should be 16 for PCM
        self.format_data_length = int.from_bytes(self.h[16:20], 'little')
        # 1 indicates PCM format
        self.format_type = int.from_bytes(self.h[20:22], 'little') 
        if not(self.format_type == 1):
            print("Not PCM coded")
            return False
        self.n_channels = int.from_bytes(self.h[22:24], 'little')
        self.fs = int.from_bytes(self.h[24:28], 'little')
        self.byte_rate = int.from_bytes(self.h[28:32], 'little')        
        self.block_align = int.from_bytes(self.h[32:34], 'little')
        self.bit_depth = int.from_bytes(self.h[34:36], 'little')
        if not(self.byte_rate == self.fs*self.bit_depth*self.n_channels/8):
            print("Byte rate not conforming to fs, bit depth and number of channels")
            return False
        chunk_type_b = self.file.read(4)
        while not(chunk_type_b.decode() == 'data'):
            chunk_length = int.from_bytes(self.file.read(4), 'little')
            self.file.seek(self.file.tell()+chunk_length)
            chunk_type_b = self.file.read(4)
        # According to e.g.
        # https://sites.google.com/site/musicgapi/technical-documents/wav-file-format
        # there might be extra format bytes here. Worth checking in case of errors.
#        self.data_start = chunk_type_b # Indicate start of data info fields
        chunk_length_b = self.file.read(4)
        self.h = self.h + chunk_type_b + chunk_length_b
        self.data_length = int.from_bytes(chunk_length_b, 'little')
        self.data_start_byte = self.file.tell()
        return True
    
    def __parse_smpl(self):
        self.smpl = SimpleNamespace()
        smpl_start = self.f.find(b'smpl')
        self.smpl.len = int.from_bytes(self.f[smpl_start+4:smpl_start+8], 'little')
        self.smpl.n_loops = int.from_bytes(self.f[smpl_start+0x24:smpl_start+0x24+4], 'little')
        self.smpl.loops = []
        for i in range(self.smpl.n_loops):
            loop_start = smpl_start + 0x2C + 24*i
            lb = self.f[loop_start:loop_start+24]
            loop = SimpleNamespace()
            loop.cue_point_id = int.from_bytes(lb[0:4], 'little')
            loop.type = int.from_bytes(lb[4:8], 'little')
            loop.start = int.from_bytes(lb[8:12], 'little')
            loop.end = int.from_bytes(lb[12:16], 'little')
            loop.fraction = int.from_bytes(lb[16:20], 'little')
            loop.play_count = int.from_bytes(lb[20:24], 'little')
            self.smpl.loops.append(loop)
        
    def _create_loop(self, loop_start, loop_end):
        smpl = b'smpl'
        n_loops = 1
        n_loops_b = n_loops.to_bytes(4, 'little')
        smpl_len = 36 + 24*n_loops
        smpl_len_b = smpl_len.to_bytes(4, 'little')
        ef = (0).to_bytes(4, 'little') # empty field
        loop_start_b = loop_start.to_bytes(4, 'little')
        loop_end_b = (loop_end-1).to_bytes(4, 'little')
        loop_b = ef + ef + loop_start_b + loop_end_b + ef + ef
        smpl_b = b'smpl' + smpl_len_b + ef*7 + n_loops_b + ef + loop_b
        return smpl_b
    
    def __read_footer(self):
        self.f = self.file.read()
        if b'smpl' in self.f:
            self.__parse_smpl()
        if b'cue ' in self.f:
            self.cue = SimpleNamespace()
            self.cue_start = self.f.find(b'cue ')
        
    def __read_data(self):
        self.file.seek(self.data_start_byte)
        self._data = self.file.read(self.data_length)
        
    def write_with_loop(self, fname, loop_start, loop_end):
        smpl_b = self._create_loop(loop_start, loop_end)
        file_size = 44 - 8 + len(self._data) + len(smpl_b)
        file_size_b = file_size.to_bytes(4, 'little')
        with open(fname,'wb') as f:
            f.write(b'RIFF')
            f.write(file_size_b)
            f.write(self.h[8:])
            f.write(self._data)
            f.write(smpl_b)
        
    @property
    def data(self):
        if not(self.bit_depth in [8, 16]):
            print("WavInterface.data only working for bit depth 8 or 16, returning None")
            return None
        np_type = {8: np.int8, 16: np.int16, 24: 'no np_type'}[self.bit_depth]
        return np.frombuffer(self._data, np_type)   