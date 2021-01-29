import os
import pickle

import numpy as np
import tensorflow as tf
from numpy.random import choice

from utils import SelfAttention, piano_roll_to_pretty_midi


def generate_notes(composer, unique_notes, max_generated=1000, seq_len=50, notes=None):
    if not notes:
        notes = ['35']

    generated = [note_tokenizer.notes_to_index['e'] for i in range(seq_len - len(notes))]
    generated += [note_tokenizer.notes_to_index[note] for note in notes]

    for i in range(max_generated):
        test_input = np.array([generated])[:, i:i + seq_len]
        predicted_note = composer.predict(test_input)
        random_note_pred = choice(unique_notes + 1, 1, replace=False, p=predicted_note[0])
        generated.append(random_note_pred[0])

    return generated


def write_midi(generate, midi_file_name="generated_song.mid", start_index=49,
               fs=8, max_generated=1000):
    note_string = [note_tokenizer.index_to_notes[ind_note] for ind_note in generate]
    array_piano_roll = np.zeros((128, max_generated + 1), dtype=np.int16)

    for index, note in enumerate(note_string[start_index:]):
        if note == 'e':
            continue
        splitted_note = note.split(',')
        for j in splitted_note:
            array_piano_roll[int(j), index] = 1

    generate_to_midi = piano_roll_to_pretty_midi(array_piano_roll, fs=fs)
    for note in generate_to_midi.instruments[0].notes:
        note.velocity = np.random.randint(50, 100)
    generate_to_midi.write(os.path.join('..', 'classical_composer_data', 'generated_songs', midi_file_name))

    print(midi_file_name, ' generated.')


if __name__ == '__main__':
    author = 'JohannSebastianBach'
    weights = 'm2_epoch200.h5'
    generated_name = 'jsb2_200.mid'

    model_dir = os.path.join(os.path.dirname(__file__), '..', 'classical_composer_data', 'models', author)

    composer = tf.keras.models.load_model(os.path.join(model_dir, weights),
                                          custom_objects=SelfAttention.get_custom_objects())
    note_tokenizer = pickle.load(open(os.path.join(model_dir, "tokenizer.p"), "rb"))

    max_generate = 500
    unique_notes = note_tokenizer.unique_word
    seq_len = 50
    generated = generate_notes(composer, unique_notes, max_generate, seq_len)
    write_midi(generated, generated_name, fs=5)
