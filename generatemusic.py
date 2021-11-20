import tensorflow.keras as keras
import json
import numpy as np
import music21 as m21


class MusicGenerator:
    def __init__(self, path_of_model="save_model.h5"):
        self.path_of_model = path_of_model
        self.lstm_model = keras.models.load_model(path_of_model)

        with open("mapping/map.json", "r") as mapping_file:
            self.model_mappings = json.load(mapping_file)

        self.song_start_indicators = ["/"] * 64

    def sample_with_temp(self, probs, tmp):
        """
        Samples an index out of probs using the tmp value
        """
        preds = np.log(probs) / tmp
        probs = np.exp(preds) / np.sum(np.exp(preds))

        c = range(len(probs))
        index = np.random.choice(c, p=probs)
        return index

    def convert_music(self, in_music, step_dur=0.25, in_format="mid", file_name="output_music.midi"):
        """
        :param in_music: Music string we have created by generate_music function
        :param step_dur: Step duration of the music
        :param in_format: File format of the output music file
        :param file_name: Name of the output music file
        """

        m21_stream = m21.stream.Stream()

        """
        We have the notes in numbered formats we have been using such as 74 _ _ _ . We need to convert the numbers
        to the notes, and the underscores to the duration of the notes.
        """
        step_ctr = 1
        start_sym = None

        for i, sym in enumerate(in_music):
            if sym != "_" or i + 1 == len(music):
                if start_sym is not None:
                    quarter_dur = step_dur * step_ctr

                    if start_sym == "r":
                        e = m21.note.Rest(quarterLength=quarter_dur)
                    else:
                        e = m21.note.Note(int(start_sym), quarterLength=quarter_dur)

                    m21_stream.append(e)
                    step_ctr = 1
                start_sym = sym
            else:
                step_ctr += 1

        m21_stream.write(in_format, file_name)

    def generate_music(self, seed, number_steps, max_sequence_size, tmp):
        """
        :param seed: A piece of music we want to input as a seed so that the network can continue that seed
        :param number_steps: Time series representation we want from the network to generate
        :param max_sequence_size: This tells the network how many steps we want to consider in the seed for the network
        :param tmp: A float value that can have a range 0-1. By using probability distribution, we determine
        the way we sample the output symbols
        :return:
        """

        """
            First, we must create the seed with the start symbols
        """
        seed = seed.split()
        init_music = seed
        seed = self.song_start_indicators + seed

        """
            We are reading the seed list that has keys in the lookup table and map that onto relative integers
        """
        seed = [self.model_mappings[symbol] for symbol in seed]

        for i in range(number_steps):
            seed = seed[-max_sequence_size:]  # Limit the seed so that it doesn't exceed the max_sequence_length

            """
            Use one-hot encoding to encode the seed
            """
            encoded_seed = keras.utils.to_categorical(seed, num_classes=len(self.model_mappings))

            """
                Adds extra dimension and leaves other stuff from the seed
            """
            encoded_seed = encoded_seed[np.newaxis, ...]

            """
                Make predictions
            """
            probs = self.lstm_model.predict(encoded_seed)[0]

            output_int = self.sample_with_temp(probs, tmp)

            """
                We made prediction, add the prediction to the seed.
            """
            seed.append(output_int)

            """
                Encode the int returned as output_int
            """
            output_sym = [k for k, v in self.model_mappings.items() if v == output_int][0]

            if output_sym == "/":
                break

            init_music.append(output_sym)

        return init_music


if __name__ == "__main__":
    mg = MusicGenerator()
    seed = "55 _ 60 _ 60 _ 60 _ 62 _ 64 _ 62 _ 60 _ _ _ 64 _ 64 _ 64 _ 65 _ 67 _ _ 65 64 _ 60 _ 72 _ _ _ 72"
    music = mg.generate_music(seed, 500, 96, 0.3)
    mg.convert_music(music)
