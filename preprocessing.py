import glob
import os

import numpy as np
import pandas as pd
import pretty_midi


class NoteTokenizer:

    def __init__(self):
        self.notes_to_index = {}
        self.index_to_notes = {}
        self.notes_freq = {}
        self.num_of_word = 0
        self.unique_word = 0

    def partial_fit(self, notes):
        """ Partial fit on the dictionary of the tokenizer

        Parameters
        ==========
        notes : list of notes

        """

        for note in notes:
            note_str = ','.join(str(a) for a in note)
            if note_str in self.notes_freq:
                self.notes_freq[note_str] += 1
                self.num_of_word += 1
            else:
                self.notes_freq[note_str] = 1
                self.unique_word += 1
                self.num_of_word += 1
                self.notes_to_index[note_str], self.index_to_notes[self.unique_word] = self.unique_word, note_str

    def transform(self, list_array):
        """ Transform a list of note in string into index.

        Parameters
        ==========
        list_array : list
            list of note in string format

        Returns
        =======
        The transformed list in numpy array.

        """

        transformed_list = []
        for instance in list_array:
            transformed_list.append([self.notes_to_index[note] for note in instance])
        return np.array(transformed_list, dtype=np.int32)

    def add_new_note(self, note):
        """ Add a new note into the dictionary

        Parameters
        ==========
        note : str
            a new note who is not in dictionary.

        """

        assert note not in self.notes_to_index
        self.unique_word += 1
        self.notes_to_index[note], self.index_to_notes[self.unique_word] = self.unique_word, note


def generate_dict_time_notes(list_all_midi, batch_song=16, start_index=0, fs=30):
    """ Generate map (dictionary) of music ( in index ) to piano_roll (in np.array)

    Parameters
    ==========
    list_all_midi : list
        List of midi files
    batch_music : int
      A number of music in one batch
    start_index : int
      The start index to be batched in list_all_midi
    fs : int
      Sampling frequency of the columns, i.e. each column is spaced apart
        by ``1./fs`` seconds.

    Returns
    =======
    dictionary of music to piano_roll (in np.array)

    """

    assert len(list_all_midi) >= batch_song

    dict_time_notes = {}

    for i in range(start_index, min(start_index + batch_song, len(list_all_midi))):
        midi_file_name = list_all_midi[i]
        try:
            midi_pretty_format = pretty_midi.PrettyMIDI(midi_file_name)
            piano_midi = midi_pretty_format.instruments[0]  # Get the piano channels
            piano_roll = piano_midi.get_piano_roll(fs=fs)
            dict_time_notes[i] = piano_roll
        except Exception as e:
            print(e)
            print("broken file : {}".format(midi_file_name))
            pass

    return dict_time_notes


def process_notes_in_song(dict_time_notes, seq_len=50):
    """
    Iterate the dict of piano rolls into dictionary of timesteps and note played

    Parameters
    ==========
    dict_time_notes : dict
        dict contains index of music ( in index ) to piano_roll (in np.array)
    seq_len : int
        Length of the sequence

    Returns
    =======
    Dict of timesteps and note played

    """

    list_of_dict_keys_time = []

    for key in dict_time_notes:
        sample = dict_time_notes[key]
        times = np.unique(np.where(sample > 0)[1])
        index = np.where(sample > 0)
        dict_keys_time = {}

        for time in times:
            index_where = np.where(index[1] == time)
            notes = index[0][index_where]
            dict_keys_time[time] = notes

        list_of_dict_keys_time.append(dict_keys_time)

    return list_of_dict_keys_time


def generate_input_and_target(dict_keys_time, seq_len=50):
    """ Generate input and the target of our deep learning for one music.

    Parameters
    ==========
    dict_keys_time : dict
         Dictionary of timestep and notes
    seq_len : int
        The length of the sequence

    Returns
    =======
    Tuple of list of input and list of target of neural network.

    """

    start_time, end_time = list(dict_keys_time.keys())[0], list(dict_keys_time.keys())[-1]
    list_training, list_target = [], []

    for index_enum, time in enumerate(range(start_time, end_time)):
        list_append_training, list_append_target = [], []
        start_iterate = 0

        if index_enum < seq_len:
            start_iterate = seq_len - index_enum - 1
            for i in range(start_iterate):
                list_append_training.append('e')

        for i in range(start_iterate, seq_len):
            index_enum = time - (seq_len - i - 1)
            if index_enum in dict_keys_time:
                list_append_training.append(','.join(str(x) for x in dict_keys_time[index_enum]))
            else:
                list_append_training.append('e')

        if time + 1 in dict_keys_time:
            list_append_target.append(','.join(str(x) for x in dict_keys_time[time + 1]))
        else:
            list_append_target.append('e')

        list_training.append(list_append_training)
        list_target.append(list_append_target)

    return list_training, list_target


def generate_batch_song(list_all_midi, batch_music=16, start_index=0, fs=30, seq_len=50):
    """
    Generate Batch music that will be used to be input and output of the neural network

    Parameters
    ==========
    list_all_midi : list
        List of midi files
    batch_music : int
        A number of music in one batch
    start_index : int
        The start index to be batched in list_all_midi
    fs : int
        Sampling frequency of the columns, i.e. each column is spaced apart
        by ``1./fs`` seconds.
    seq_len : int
        The sequence length of the music to be input of neural network

    Returns
    =======
    Tuple of input and target neural network

    """

    assert len(list_all_midi) >= batch_music
    dict_time_notes = generate_dict_time_notes(list_all_midi, batch_music, start_index, fs)

    list_musics = process_notes_in_song(dict_time_notes, seq_len)
    collected_list_input, collected_list_target = [], []

    for music in list_musics:
        list_training, list_target = generate_input_and_target(music, seq_len)
        collected_list_input += list_training
        collected_list_target += list_target

    return collected_list_input, collected_list_target


def get_sampled_midi(composer='', size=100):
    if composer != '':
        midi_csv = pd.read_csv('../classical_composer_data/maestro-v1.0.0/maestro-v1.0.0.csv')
        midi_csv = midi_csv[midi_csv['canonical_composer'] == composer]
        midi_csv.reset_index(inplace=True)
        midi_names = midi_csv['midi_filename']

        sampled_midi = glob.glob(os.path.join('../classical_composer_data/maestro-v1.0.0/', midi_names[0]))
        for name in midi_names:
            sampled_midi += glob.glob(os.path.join('../classical_composer_data/maestro-v1.0.0/', name))
        sampled_midi = sampled_midi[1:]

    else:
        list_all_midi = glob.glob('../classical_composer_data/maestro-v1.0.0/**/*.midi')
        sampled_midi = list_all_midi[0:size]

    return sampled_midi
