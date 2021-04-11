# sampling-tools
Tools for automating various tasks involved in creating sample-based instruments. 

## Usage
The basic use-case for this toolchain is to create a sample-based instruments from a MIDI-controlled synthesizer:
1. Use auto-sampling_add_notes_and_regions.lua to prepare a Reaper project for recording the instrument
2. Record the instrument and render the samples (see instructions for the script below), or simply render if you want to use a VST instrument
3. Use loop_samples.py to find loop points in the samples and create loopable files
4. Use sfz_from_dir.py to create a .sfz mapping of the samples which can be loaded into a sampler

### Example of Using loop_samples.py and sfz_from_dir.py
An example has been provided which demonstrates using loop_samples.py to create loopable .wav files from raw input samples, followed by using sfz_from_dir.py to create a .sfz mapping which can be used in e.g. sforzando (https://www.plogue.com/products/sforzando.html) or imported in Kontakt. The ReaScript is not included in this example but has been used to create the raw samples used as input. First run:
```
python loop_samples.py -cfg example_config.json
```
This will create a timestamped output directory in the example_input folder, the looped samples can be found in the looped_samples subidrectory. The diagnostics directory contains plots of the output around the loop point and .wav clips with the loop point halfway into the file, listen to these to make sure there is no click or similar artifact due to the looping. The example samples all have low pitch which makes the search for loop points fairly quick. If you try using a sample with higher pitch you will find that this takes longer due to higher number of zero crossings (see below for more details).

Once you have a set of looped samples that you are satisfied with, copy these into ./my_looped_samples and run:
```
python sfz_from_dir.py -s my_looped_samples -o my_sfz_instrument.sfz
```
This will create the file ./my_sfz_instrument.sfz which contains a mapping that can be used to load the samples in ./my_looped_samples into a sampler. The path is relative so if you want to move the instrument somewhere else you will need to move both the .sfz file and the directory with the samples. .sfz files are plaintext so it is easy to edit the file to make the sampler look for the samples in a different directory.

## Descriptions of Individual Scripts

### auto-sampling_add_notes_and_regions.lua
ReaScript to add MIDI notes and regions for automatic sampling of e.g. external instruments of VSTs. Use Actions > Show Action List > New Action > Load ReaScript to load the script into Reaper. Click the Edit Action button to edit the script. The parameters (which interval to use for sampling, length of each note etc.) are set at the top of the script. The latency_offset_length parameter might be superfluous as you can easily compensate for this after recording by zooming in and dragging the audio item.

After after adding notes/regions and recording (or adding a VST for output), select File > Render. Set Bounds to Project regions and File name to $region, this will export         each region to a file with the naming scheme:<br>
<name_prefix>\_\<root note>\_\<low key>\_\<high key>\_\<min velocity>\_\<max velocity>\_\<group name><br>
See e.g. https://www.adsrsounds.com/kontakt-tutorials/how-to-auto-map-samples-in-kontakt/ for instructions about how to use these names for automapping in Kontakt (a useful addition would be to add code for generating an .sfz file with mappings).

### loop_samples.py
Look for appropriate loop points in a .wav file and write the result to a new file, either by truncating the waveform so only the loop is kept or by adding loop points to the new file as metadata which can be used by a sampler to determine where to loop. The algorithm looks compares the neighborhood of zero-crossings in the file and attempts to find two crossings where the neighborhoods have a small L2 distance. The segment to look in can be set using the "Loop start" and "Loop end" arguments, only zero-crossings between these will be examined. The "Min loop length" arguments sets a minimum length of the resulting loop, this has two purposes:
* Avoiding short loops as these can easily lead to audible repetitiveness
* Limiting the amount of zero-crossing to examine, if the minimum loop length is close to the distance between the loop start and end few zero crossings will fulfill the requirement on loop length which speeds up the search process
Using "Min loop length" to limit the number of crossings to examine is especially important for samples which has lots of crossings, e.g. samples with high pitch, detune or noise. Multiprocessing is used to speed up the search process, the number of parallel processes is set using the "N workers" argument which should usually be set to equal the number of CPU cores on the machine or one lower if you wish to keep one core unloaded for other tasks.
```
python loop_samples.py -h
```
to get a list of available arguments.

### sfz_from_dir.py
Create a .sfz file from the contents of a directory, all files must have the expected naming scheme which can be set using the -f argument. Use -h to see arguments with hints for values.

 
