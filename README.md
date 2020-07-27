# sampling-tools
Tools for automating various tasks involved in creating sample-based instruments. 

## auto-sampling_add_notes_and_regions
ReaScript to add MIDI notes and regions for automatic sampling of e.g. external instruments of VSTs. Use Actions > Show Action List > New Action > Load ReaScript to load the script into Reaper. Click the Edit Action button to edit the script. The parameters (which interval to use for sampling, length of each note etc.) are set at the top of the script. The latency_offset_length parameter might be superfluous as you can easily compensate for this after recording by zooming in and dragging the audio item.

After after adding notes/regions and recording (or adding a VST for output), select File > Render. Set Bounds to Project regions and File name to $region, this will export         each region to a file with the naming scheme:<br>
<name_prefix>\_\<root note>\_\<low key>\_\<high key>\_\<min velocity>\_\<max velocity>\_\<group name><br>
See e.g. https://www.adsrsounds.com/kontakt-tutorials/how-to-auto-map-samples-in-kontakt/ for instructions about how to use these names for automapping in Kontakt (a useful addition would be to add code for generating an .sfz file with mappings).

## sfz_from_dir
Create a .sfz file from the contents of a directory, all files must have the expected naming scheme which can be set using the -f argument. Use -h to see arguments with hints for values.
