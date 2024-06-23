import string
from collections import defaultdict

# morphology rules used to assign unknown word tokens
noun_suffix = ["اله", "باز", "بان", "‌تر", "سار", "ستان", "سرا", "سیر", "کار", "گاه", "ناک", "مند", "نده", "وار", "گون","دان"]
verb_suffix = ["‌ام", "‌اید", "‌اند", "‌ایم"]
adv_suffix = ["آسا", "آگین", "سان"]
adj_suffix = ["انه", "گان", "گون"]
#noun_suffix = ["action", "age", "ance", "cy", "dom", "ee", "ence", "er", "hood", "ion", "ism", "ist", "ity", "ling", "ment", "ness", "or", "ry", "scape", "ship", "ty"]
#verb_suffix = ["ate", "ify", "ise", "ize"]
#adj_suffix = ["able", "ese", "ful", "i", "ian", "ible", "ic", "ish", "ive", "less", "ly", "ous"]
#adv_suffix = ["ward", "wards", "wise"]

def get_word_tag(line, vocab):
    if not line.split(): #then the line is empty
        word = "--n--"
        tag = "--s--"
        return word, tag
    else:
        word, tag = line.split()
        if word not in vocab:
            word = assign_unkown(word) #unkown word
        return word, tag
    return None

def processing(vocab, text):
    prep_sentence = []
    for word in text:
        if not word.split(): #empty
            word = "--n--"
            prep_sentence.append(word)
            continue
        elif word.strip() not in vocab:
            word = assign_unkown(word) #word
            prep_sentence.append(word)
            continue
        else:
            prep_sentence.append(word.strip())
    assert(len(prep_sentence) == len(text)) #same size
    return prep_sentence

def assign_unkown(tok):

    if str(tok).isascii(): #then it's english:
        noun_suffix = ["action", "age", "ance", "cy", "dom", "ee", "ence", "er", "hood", "ion", "ism", "ist", "ity", "ling", "ment", "ness", "or", "ry", "scape", "ship", "ty"]
        verb_suffix = ["ate", "ify", "ise", "ize"]
        adj_suffix = ["able", "ese", "ful", "i", "ian", "ible", "ic", "ish", "ive", "less", "ly", "ous"]
        adv_suffix = ["ward", "wards", "wise"]
    else:
        noun_suffix = ["اله", "باز", "بان", "‌تر", "سار", "ستان", "سرا", "سیر", "کار", "گاه", "ناک", "مند", "نده",
                       "وار", "گون", "دان"]
        verb_suffix = ["‌ام", "‌اید", "‌اند", "‌ایم"]
        adv_suffix = ["آسا", "آگین", "سان"]
        adj_suffix = ["انه", "گان", "گون"]
    # Digits
    if any(char.isdigit() for char in tok):
        return "--unk_digit--"

    # Punctuation
    elif any(char in set(string.punctuation) for char in tok):
        return "--unk_punct--"

    # Upper-case
    elif any(char.isupper() for char in tok):
        return "--unk_upper--"

    # Nouns
    elif any(tok.endswith(suffix) for suffix in noun_suffix):
        return "--unk_noun--"

    # Verbs
    elif any(tok.endswith(suffix) for suffix in verb_suffix):
        return "--unk_verb--"

    # Adjectives
    elif any(tok.endswith(suffix) for suffix in adj_suffix):
        return "--unk_adj--"

    # Adverbs
    elif any(tok.endswith(suffix) for suffix in adv_suffix):
        return "--unk_adv--"

    return "--unk--"

def build_vocab(corpus_path):
    with open(corpus_path, 'r') as f:
        lines = f.readlines()

    tokens = [line.split('\t')[0] for line in lines]
    freqs = defaultdict(int)
    for tok in tokens:
        freqs[tok] += 1

    vocab = [k for k, v in freqs.items() if (v > 1 and k != '\n')]
    unk_toks = ["--unk--", "--unk_adj--", "--unk_adv--", "--unk_digit--", "--unk_noun--", "--unk_punct--", "--unk_upper--", "--unk_verb--"]
    vocab.extend(unk_toks)
    vocab.append("--n--")
    vocab.append(" ")
    vocab = sorted(set(vocab))
    return vocab
