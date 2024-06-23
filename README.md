# POS-HMM-Farsi
Part-Of-Speech tagging persian and english sentences using a HMM (Hidden Markov Model) that utilizes Viterbi Algorithm. The Farsi dataset uses a modified and normalized Bijankhan dataset.

# How to use:
First run train.py and choose the dataset. It'll generate and serilize emission and transition weights files. Then run pos.py and choose the dataset that it was trained at the first step. Then write your sentence in a formal way. It'll guess the most probable sequence of POS tags that fits the sentence the best.
Refer to Bijankhan-tagset-description.txt file for pos taggs for the Farsi dataset.

# TODO:
- Improve UI and add arg parsing
- Add a jupyter notebook for explaing the viterbi algorithm
