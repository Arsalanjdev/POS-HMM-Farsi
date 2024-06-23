import pickle
from numpy import save
from hmm import build_vocab_to_index, create_dictionaries, create_transition_matrix, create_emission_matrix
from hmm import load_training_data

def load_data():
    dataset_file = "eng.pos"
    type = input("Choose your dataset: Enter f for farsi dataset,"
                 "Enter e for English dataset: ").lower()
    if type == 'f':
        dataset_file = 'farsi.pos'
    vocabulary_to_index = build_vocab_to_index(dataset_file)
    f = open('vocab.pkl', 'wb')
    pickle.dump(vocabulary_to_index, f)
    f.close()

    training_corpus = load_training_data(dataset_file)
    emission_counts, transition_counts, tag_counts = create_dictionaries(training_corpus, vocabulary_to_index)
    states = sorted(tag_counts.keys())
    smoothing_factor = 0.001
    transmition_matrix = create_transition_matrix(transition_counts, tag_counts, smoothing_factor)
    emission_matrix = create_emission_matrix(emission_counts, tag_counts, list(vocabulary_to_index), smoothing_factor)
    save('transition_matrix.npy', transmition_matrix)
    save('emission_matrix.npy', emission_matrix)


if __name__ == "__main__":
    load_data()
