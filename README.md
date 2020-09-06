# POS-Tagger

This program implements hidden markov models, the viterbi algorithm, and nested maps to tag parts of speech in text files. Since the same word can serve as different parts of speech in different contexts, the hidden markov model keeps track of log-probabilities for a word being a particular part of speech(observation score) as well as a part of speech being followed by another part of speech(transition score). A simple unit test driver for the Viterbi algorithm is also included.

### To Run
If the `test` boolean in main is set to true, the tagger will be tested on a pair of test files, which are selected by the user. If the `input` boolean in main is set to true, sentences input from the console will be tagged.

