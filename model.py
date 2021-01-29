import os
import pickle
from random import shuffle

import keras
import numpy as np
import tensorflow as tf
from keras.layers import Dense, Dropout, Embedding, GRU, Input, ReLU
from keras.losses import sparse_categorical_crossentropy
from keras.models import Model
from keras.optimizers import Adam

from preprocessing import NoteTokenizer, generate_batch_song, generate_dict_time_notes, get_sampled_midi, \
    process_notes_in_song
from utils import SelfAttention


class ClassicalComposer:
    def __init__(self, epochs, note_tokenizer, sampled_midi, fs, batch_size, batch_song, optimizer,
                 loss_fn, total_songs, seq_len, unique_notes, directory):

        self.epochs = epochs
        self.note_tokenizer = note_tokenizer
        self.sampled_midi = sampled_midi
        self.fs = fs
        self.batch_size = batch_size
        self.batch_song = batch_song
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.total_songs = total_songs
        self.seq_len = seq_len
        self.directory = directory
        self.model = self.get_model(seq_len, unique_notes)

    def get_model(self, seq_len, unique_notes, dropout=0.3, output_embedding=128, gru_unit=96,
                  dense_unit=64):
        inputs = Input(shape=(seq_len,), name='input')
        x = Embedding(input_dim=unique_notes + 1, output_dim=output_embedding,
                      input_length=seq_len, name='embedding')(inputs)
        x, _ = SelfAttention(return_attention=True,
                             attention_activation='sigmoid',
                             attention_type=SelfAttention.ATTENTION_TYPE_MUL,
                             attention_width=128,
                             kernel_regularizer=keras.regularizers.l2(1e-4),
                             bias_regularizer=keras.regularizers.l1(1e-4),
                             attention_regularizer_weight=1e-4,
                             name='self_attention_1')(x)
        x = Dropout(dropout, name='dropout_1')(x)
        x, _ = SelfAttention(return_attention=True,
                             attention_activation='sigmoid',
                             attention_type=SelfAttention.ATTENTION_TYPE_MUL,
                             attention_width=128,
                             kernel_regularizer=tf.keras.regularizers.l2(1e-4),
                             bias_regularizer=tf.keras.regularizers.l1(1e-4),
                             attention_regularizer_weight=1e-4,
                             name='self_attention_2')(x)
        x = Dropout(dropout, name='dropout_2')(x)
        x = GRU(gru_unit, name='gru_1')(x)
        x = Dropout(dropout, name='dropout_3')(x)
        x = Dense(dense_unit, name='dense_1')(x)
        x = ReLU(name='re_lu')(x)
        outputs = Dense(unique_notes + 1, activation="softmax", name='dense_2')(x)

        model = Model(inputs=inputs, outputs=outputs)
        model.summary()

        return model

    def train(self):
        for epoch in range(self.epochs):
            shuffle(self.sampled_midi)
            loss_total = 0
            steps = 0
            steps_nnet = 0

            for i in range(0, self.total_songs, self.batch_song):
                steps += 1
                inputs_large, outputs_large = generate_batch_song(list_all_midi=self.sampled_midi,
                                                                  batch_music=self.batch_song,
                                                                  start_index=i, fs=self.fs,
                                                                  seq_len=self.seq_len)
                inputs_large = np.array(self.note_tokenizer.transform(inputs_large), dtype=np.int32)
                outputs_large = np.array(self.note_tokenizer.transform(outputs_large), dtype=np.int32)

                index_shuffled = np.arange(start=0, stop=len(inputs_large))
                np.random.shuffle(index_shuffled)

                for nnet_steps in range(0, len(index_shuffled), self.batch_size):
                    steps_nnet += 1
                    current_index = index_shuffled[nnet_steps:nnet_steps + self.batch_size]
                    inputs, outputs = inputs_large[current_index], outputs_large[current_index]

                    if len(inputs) // self.batch_size != 1:
                        break

                    loss = self.train_step(inputs, outputs)
                    loss_total += tf.math.reduce_sum(loss)

                    if steps_nnet % 200 == 0:
                        print("\n epochs: {} | steps: {} | total_loss={}".format(epoch + 1, steps_nnet,
                                                                                 loss_total))

            self.model.save(os.path.join(self.directory, 'm2_epoch{}.h5'.format(epoch + 1)))
            pickle.dump(note_tokenizer, open(os.path.join(self.directory, "tokenizer.p"), "wb"))

    @tf.function
    def train_step(self, inputs, targets):
        with tf.GradientTape() as tape:
            prediction = self.model(inputs)
            loss = self.loss_fn(targets, prediction)
        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

        return loss


# ===================================================================================================

if __name__ == '__main__':
    epochs = 200
    seq_len = 50
    fs = 5
    batch_size = 96
    batch_song = 16
    dir = '../classical_composer_data/models/JohannSebastianBach/'

    author = 'Johann Sebastian Bach'

    sampled_midi = get_sampled_midi(composer=author)

    note_tokenizer = NoteTokenizer()

    for i in range(len(sampled_midi)):
        dict_time_notes = generate_dict_time_notes(sampled_midi, batch_song=1, start_index=i, fs=fs)
        full_notes = process_notes_in_song(dict_time_notes)
        for note in full_notes:
            note_tokenizer.partial_fit(list(note.values()))
    note_tokenizer.add_new_note('e')

    optimizer = Adam()
    loss_fn = sparse_categorical_crossentropy

    composer = ClassicalComposer(epochs=epochs, note_tokenizer=note_tokenizer, sampled_midi=sampled_midi,
                                 fs=fs, batch_size=batch_size, batch_song=batch_song,
                                 optimizer=optimizer, loss_fn=loss_fn, total_songs=len(sampled_midi),
                                 seq_len=seq_len, unique_notes=note_tokenizer.unique_word, directory=dir)
    composer.train()
