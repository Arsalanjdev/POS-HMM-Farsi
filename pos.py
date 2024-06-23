from numpy import load
import pickle
from nltk import word_tokenize
from hmm import load_training_data
from hmm import create_dictionaries, initialize, viterbi_forward, viterbi_backward
from utils import processing


def predict():
    dataset_file = "eng.pos"
    type = input("Enter the type of dataset in which the model was trained for. e for english, f for farsi:\n").lower()
    if type == 'f':
        dataset_file = "farsi.pos"
    sentence = input("Enter the sentence you wish to determine its POS-tagging:\n")
    #sentence = "سلام من ده عدد دوست دارم."
    sample = str(sentence) + ' #'
    tokens = word_tokenize(sample)
    # print(tokens)
    file = open('vocab.pkl', 'rb')
    vocab2idx = pickle.load(file)
    file.close()
    prep_tokens = processing(vocab2idx, tokens)
    training_corpus = load_training_data(dataset_file)
    emission_counts, transition_counts, tag_counts = create_dictionaries(training_corpus, vocab2idx)
    states = sorted(tag_counts.keys())
    #create_transition_matrix(transition_counts, tag_counts, alpha)
    #create_emission_matrix(emission_counts, tag_counts, list(vocab2idx), alpha)
    transition_matrix = load('transition_matrix.npy')
    emission_matrix = load('emission_matrix.npy')
    best_probs, best_paths = initialize(transition_matrix, emission_matrix, tag_counts, vocab2idx, states, prep_tokens)
    best_probs, best_paths = viterbi_forward(transition_matrix, emission_matrix, prep_tokens, best_probs, best_paths, vocab2idx)
    pred = viterbi_backward(best_probs, best_paths, states)

    res = []
    for tok, tag in zip(prep_tokens[:-1], pred[:-1]):
        res.append((tok, tag))
    print(res)


if __name__ == "__main__":
    predict()
