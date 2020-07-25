-- Version: 0.1, 2020-07-25
-- Author: Jonas Tallhage, jonas@tallhage.se, https://github.com/jonastallhage
-- License: MIT
-- Documentation: https://github.com/jonastallhage/autosampling-reascripts

-- Usage: To run the script in reaper, select Actions > Show Action List > New Action
--        > Load ReaScript and load the file.
--        The script will add a track with MIDI notes along with regions which can
--        be used for automatically exporting and naming the samples. After adding
--        the notes and recording (or adding a VST instrument), select File > Render.
--        Set Bounds to Project regions and File name to $region, this will export
--        each region to a file with the naming scheme:
--        <name_prefix>_<root note>_<low key>_<high key>_<min velocity>_<max velocity>_<group name>
--        See e.g. https://www.adsrsounds.com/kontakt-tutorials/how-to-auto-map-samples-in-kontakt/
--        for instructions about how to use these names for automapping in Kontakt.

-- USER INPUT
note_nbr_low = 40 -- lowest MIDI note number
note_nbr_high = 90 -- highest MIDI note number
sampling_interval = 4 -- semitone interval between samples
keyrange_down = 1 -- number of semitones down from sampled key in keyrange
keyrange_up = 2 -- number of semitones up from sampled key in keyrange
note_length = 0.1 -- Length of "key press" (seconds)
-- Velocities, note is the velocity of the MIDI event, low and high are used in the sample name (for automapping)
--velocities = {{["note"] = 40,  ["low"] =  0, ["high"] =  55},
--              {["note"] = 70,  ["low"] = 56, ["high"] =  85},
--              {["note"] = 110, ["low"] = 95, ["high"] = 127}}
velocities = {{["note"] = 90,  ["low"] =  0, ["high"] =  127}}
sample_length = 13 -- Length of sample (seconds)
pause_length = 0.5 -- Length of pause in between samples (seconds)
latency_offset_length = 0.002 -- Offset between MIDI note start and sample start to compensate for latency
name_prefix = "pulse2p491" -- Prefix string added to each sample name
N_roundrobins = 1 -- Number of roundrobin samples
t_start = 1 -- Start of first sample (seconds)



function get_midi_notes_to_letters_table()
    note_letters = {"C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"}
    midi_notes_to_letters = {}
    midi_notes_to_letters[21] = "A0"
    midi_notes_to_letters[22] = "A#0"
    midi_notes_to_letters[23] = "B0"
    midi_notes_to_letters[108] = "C8"
    for midi_note_nbr = 24 , 107, 1 do
        mod12 = (midi_note_nbr-24) % 12
        note_base_str = note_letters[mod12 + 1]
        suffix_note_nbr = math.floor((midi_note_nbr - 24)/12) + 1
        note_str = note_base_str .. tostring(suffix_note_nbr)
        midi_notes_to_letters[midi_note_nbr] = note_str
    end
    return midi_notes_to_letters
end
midi_notes_to_letters = get_midi_notes_to_letters_table()

function add_note_plus_region(note_track, 
                              note_nbr, note_velocity,                      
                              t_start, t_latency, t_note, t_sample, t_pause,
                              name_prefix, keyrange_down, keyrange_up,
                              velocity_low, velocity_high, n_roundrobin)
    -- Set time representation to seconds
    opt_bool_qnIn = false 

    -- Add region
  region_start = t_start + t_latency
  region_end = t_start + t_latency + t_sample
  root_key = midi_notes_to_letters[note_nbr]
  low_key = midi_notes_to_letters[note_nbr-keyrange_down]
    high_key = midi_notes_to_letters[note_nbr+keyrange_up]
    region_name = (name_prefix .. "_" .. root_key .. "_" .. low_key .. "_" ..
                   high_key .. "_" .. tostring(velocity_low) .. "_" ..
                   tostring(velocity_high) .. "_" .. tostring(n_roundrobin))
    reaper.AddProjectMarker(0, true, region_start, region_end, region_name, -1)   

    -- Add MIDI item to note_track
    midi_item_start = t_start
    midi_item_end = region_end
    midi_item = reaper.CreateNewMIDIItemInProj(note_track,midi_item_start,midi_item_end,opt_bool_qnIn)
    
    -- Add note to MIDI item
    midi_item_take = reaper.GetActiveTake(midi_item)
    note_start = 0
  note_end = t_note * get_s_to_ppq_scaling()
  midi_channel = 1
    ok = reaper.MIDI_InsertNote(midi_item_take, false, false, note_start, note_end,
                                midi_channel-1, note_nbr, note_velocity,
                                false)  
end

function add_midi_track()
  reaper_proj = 0 --0 indicates current project
  track_idx = 0 -- Todo: Get an index by checking existing
  reaper.InsertTrackAtIndex(track_idx, false)
  return reaper.GetTrack(reaper_proj, track_idx)
end

function get_s_to_ppq_scaling()
    beat_to_ppq_scaling = 960 -- ppq/beat
    tempo_bpm = reaper.Master_GetTempo() -- beats/(60 s)
    tempo_bps = tempo_bpm/60
    seconds_per_beat = 1/tempo_bps --beats/s
    s_to_ppq_scaling = 4 * beat_to_ppq_scaling * seconds_per_beat -- (ppq/beat)*(beats/s) = ppq/s
    -- Note, this works but the calculation probably isn't accurate, or some input assumption is wrong
    return s_to_ppq_scaling
end

function add_notes(note_track, t_start,
                    note_nbr_low, note_nbr_high,
                    sampling_interval, keyrange_down, keyrange_up,
                    t_note, t_sample, t_pause, t_latency,
                    velocities,
                    N_roundrobins,
                    name_prefix)
    n_note = 0
    t_step = t_sample + t_pause + t_latency
    for n_roundrobin = 0, N_roundrobins-1, 1 do
        for note_nbr = note_nbr_low, note_nbr_high, sampling_interval do
            for i,velocity in pairs(velocities) do
                note_data = {}
                note_data["note_nbr"] = note_nbr
                note_data["velocity"] = velocity[3]
                note_velocity = velocity["note"]
                velocity_low = velocity["low"]
                velocity_high = velocity["high"]
                add_note_plus_region(note_track, 
                              note_nbr, note_velocity,
                              t_start, t_latency, t_note, t_sample, t_pause,
                              name_prefix, keyrange_down, keyrange_up,
                              velocity_low, velocity_high, n_roundrobin)
                t_start = t_start + t_step
            end
        end
    end
end

note_track = add_midi_track()
add_notes(note_track, t_start,
           note_nbr_low, note_nbr_high,
           sampling_interval, keyrange_down, keyrange_up,
           note_length, sample_length, pause_length, latency_offset_length,
           velocities,
           N_roundrobins,
           name_prefix)
