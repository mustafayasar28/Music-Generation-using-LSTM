import json

import numpy as np
import tensorflow.keras as keras

FINAL_DIRECTORY = 'final_dir'
MODEL_PATH = "save_gru_model.h5"
NUMBER_OF_OUTPUT_UNITS = 38
LOSS = "sparse_categorical_crossentropy"
LEARNING_RATE = 0.001
NUMBER_OF_NEURAL_UNITS = [256]
EPOCHS = 50
BATCH_SIZE = 64  # Amount of sample that the networks will see before running back propagation


def generate_sequence():
    with open(FINAL_DIRECTORY + "/final", "r") as final_file:
        final_song = final_file.read()

    final_song_map = final_song.split()

    with open("mapping/map.json", "r") as mp_file:
        mapping_dict = json.load(mp_file)

    """
     Now that we have encoded our songs, now we will convert the songs into integers so that the 
     model can train on them
    """
    int_repr_songs = []
    for key in final_song_map:
        int_repr_songs.append(mapping_dict[key])

    """
        In order to create a neural network, we are going to get the subsequences of time series of songs.
        Then we will feed the network with those subsequences, and their relative targets. 
    """
    model_inputs = []
    model_targets = []

    len_sequence = 64
    total_num_sequences = len(int_repr_songs) - len_sequence

    """
        At every step of GRU network, there will be an input, and there will be a target.
        For instance, while guessing the 11 th element, the first 10 elements are the inputs,
        and the real 11th element is the target. The output of the LSTM at that step is recorded.
    """
    for i in range(total_num_sequences):
        model_inputs.append(int_repr_songs[i: i + len_sequence])
        model_targets.append(int_repr_songs[i + len_sequence])

    """
        To prepare the input and output, one-hot encoding of the sequences is used.    
        To understand why: https://machinelearningmastery.com/why-one-hot-encode-data-in-machine-learning/ 
    """
    size_vocab = len(set(int_repr_songs))
    model_inputs = keras.utils.to_categorical(model_inputs, num_classes=size_vocab)
    model_targets = np.array(model_targets)
    return model_inputs, model_targets


def build_model(output_units, num_units, loss, learning_rate):
    """
    :param output_units: Number of output units
    :param num_units: Number of neural units
    :param loss: The type of the loss
    :param learning_rate: The learning rate of the model
    :return: model we are building
    """

    """
        First, create a windows whose length is 38 and slide it onto the songs array
    """
    gru_input = keras.layers.Input(shape=(None, output_units))

    gru_model = keras.layers.GRU(num_units[0])(gru_input)

    """
        To avoid overfitting problems, use dropout technique
    """
    gru_model = keras.layers.Dropout(0.2)(gru_model)

    output_layer = keras.layers.Dense(output_units, activation="softmax")(gru_model)

    final_model = keras.Model(gru_input, output_layer)
    final_model.compile(loss=loss, optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
                        metrics=["accuracy"])

    final_model.summary()

    return final_model


def train(output_units=NUMBER_OF_OUTPUT_UNITS, num_units=NUMBER_OF_NEURAL_UNITS, loss=LOSS,
          learning_rate=LEARNING_RATE):
    m_inputs, m_targets = generate_sequence()
    model = build_model(output_units, num_units, loss, learning_rate)
    model.fit(m_inputs, m_targets, epochs=EPOCHS, batch_size=BATCH_SIZE)
    model.save(MODEL_PATH)


if __name__ == "__main__":
    train()
