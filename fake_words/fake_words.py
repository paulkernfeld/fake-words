# -*- coding: utf-8 -*-
import codecs
import itertools
import math
import heapq
from collections import defaultdict
from collections import namedtuple
from math import e
from operator import itemgetter
from os.path import join
from random import randint, random

import numpy as np
from numpy.random import choice


BASE_PATH = "bower_components"


"""
An abstraction for language-specific information:
* (str) The name of the language
* (str) The path to a plaintext newline-separated dictionary for the language
* (str) The encoding of the dictionary
* (str) The path to a text corpus for the language
* (unicode) The alphabet of the language, listing accented characters separately
"""
LanguageInfo = namedtuple("LanguageInfo", ["name", "dict_path", "dict_encoding", "corpus_path", "alphabet"])

"""
"""
LANGUAGES = {
    "english": LanguageInfo("english", "TWL06/index.txt", "utf-8","tale/index.txt", u"abcdefghijklmnopqrstuvwxyz'"),
    "french": LanguageInfo("french", "ods5/index.txt", "utf-8","lesmis/index.txt", u"aàâäbcçdeéèêëfghiîïjklmnoöôpqrstuûüùvwxyz'"),
    "german": LanguageInfo("german", "derewo/derewo-v-30000g-2007-12-31-0.1", "iso-8859-1","siddhartha/index.txt", u"aäbcdefghijklmnoöpqrsßtuüvwxyz"),
    "italian": LanguageInfo("italian", "zingarelli2005/index.txt", "utf-8","viva/index.txt", u"aàbcdeèéfghiìlmnoòpqrstuùvz'"),
    "latin": LanguageInfo("latin", "whitaker/index", "utf-8","cicero/index.txt", u"abcdefghilmnopqrstuvwxz"),
    "polish": LanguageInfo(
        "polish",
        "ispell_pl/polish.sup",
        "iso-8859-2",
        "pan-tadeusz/index.txt",
        u"aąbcćdeęfghilłmnńoóprsśtuxyzźż"
    ),
    "spanish": LanguageInfo(
        "spanish",
        "lemario_spanish_dict/index.txt",
        "iso-8859-1",
        "dona/index.txt",
        u"aábcdeéfghiíjklmnñoópqrstuúüvwxyz"
    ),
}

# Create an alphabet that contains all letters of all alphabets
UBER_ALPHABET = set("".join([language.alphabet for language in LANGUAGES.values()]))

# The number of words to store when searching for most likely words
MAX_WORDS_TO_STORE = 1000


class Language(object):
    """
    This class represents a language. It contains all the information needed to train itself, and actually runs the
    training step in the __init__ method.
    """
    def __init__(self, info, min_gram, max_gram):
        # Build dictionary
        self.info = info
        self.dictionary = set()
        dictionary_file = codecs.open(join(BASE_PATH, info.dict_path), "r", info.dict_encoding)
        for word in dictionary_file.read().split():
            self.dictionary.add(word.lower())

        # Initialize word frequency model
        self.alphabet_with_space = self.info.alphabet + u" "

        # Train models, keeping track of strange things we find in the training data
        self.n_grams = {}
        self.found_punctuation = defaultdict(int)
        self.found_strange_letters = defaultdict(int)
        self.train_frequency_model(min_gram, max_gram)

    def generate_n_grams(self, min_gram, max_gram):
        for k in range(min_gram, max_gram + 1):
            grams = {}
            for l in itertools.product(self.info.alphabet + " ", repeat=k):
                grams[l] = 1
            self.n_grams += [grams]

    def train_frequency_model(self, min_gram, max_gram):
        # First, clean the document by replacing non-approved punctuation with a space, the start/end token
        with codecs.open(join(BASE_PATH, self.info.corpus_path), 'r', encoding='utf8') as file:
            document = list(file.read().lower())
            for i, l in enumerate(document):
                # Ignore whitespace here, it will be thrown out anyways when we split() below.
                if l.isspace():
                    continue

                # This is a normal letter, no need to do anything.
                if l in self.alphabet_with_space:
                    continue

                # Ah, a foreign letter! Replace it with a "?" as a placeholder.
                if l in UBER_ALPHABET:
                    self.found_strange_letters[l] += 1
                    document[i] = u"?"
                    continue

                # This character is punctuation; treat it the same as whitespace.
                self.found_punctuation[l] += 1
                document[i] = u" "

        # Store all n-grams
        for k in range(min_gram, max_gram + 1):
            self.n_grams[k] = defaultdict(lambda: 1)

        for word in u"".join(document).split():
            word_padded = u" {} ".format(word)

            # Record observed n-grams
            for k in range(min_gram, max_gram + 1):
                for i in range(len(word_padded) - k):
                    self.n_grams[k][tuple(word_padded[i:i+k])] += 1

        # print u"{} punctuation: {}".format(self.info.name, u" ".join(punctuation))
        # print u"{} strange letters: {}".format(self.info.name, u" ".join(strange_letters))

    def sample_progressive(self, n_times):
        sampled = set()
        for i in xrange(n_times):
            # Start with a start token
            word_so_far = ' '

            while True:
                next_probs = np.empty(len(self.alphabet_with_space))
                for i, next_letter in enumerate(self.alphabet_with_space):
                    next_probs[i] = self.get_word_prob(word_so_far + next_letter)
                next_probs = e ** next_probs
                next_letter = choice(list(self.alphabet_with_space), p=next_probs / next_probs.sum())
                word_so_far += next_letter

                # Stop the sample if we're in a terrible loop
                if len(word_so_far) >= 12:
                    word_so_far += u"..."
                    break

                # Stop the sample when we sample an end token
                if next_letter == ' ':
                    break

            sampled.add(word_so_far)

        return sampled

    def sample_mh(self, word_length, max_to_store, n_runs, n_samples):
        def pick_letter():
            return choice(list(self.info.alphabet))

        # The samples to return
        samples = []

        # A heap of the most likely words we've sampled
        top = []

        # Restart at a random word every so often
        for i in xrange(n_runs):

            # Generate an initial word
            word = [pick_letter() for _ in xrange(word_length)]
            word_prob = float("-inf")
            for j in xrange(n_samples):
                new_word = word[:]

                # Make a new word by replacing one letter...
                if random() < 0.5:
                    index_to_change = randint(0, word_length-1)
                    new_letter = pick_letter()
                    new_word[index_to_change] = new_letter

                # ...or by swapping two letters.
                else:
                    index_a = randint(0, word_length-1)
                    index_b = randint(0, word_length-1)
                    new_word[index_a], new_word[index_b] = new_word[index_b], new_word[index_a]

                new_prob = self.get_word_prob(u" {} ".format("".join(new_word)))
                if random() < e ** (new_prob - word_prob):
                    word, word_prob = new_word, new_prob

                    word_str = u"".join(word)

                    if word_str not in [top_word for top_prob, top_word in top]:
                        try_to_store(top, word_str, word_prob, max_to_store)

            # Record this sample at the end of the run
            samples.append((word_str, word_prob))

        return samples, top

    def get_word_prob(self, word):
        n_grams = self.n_grams

        word_prob = 0
        for m in range(1, len(n_grams)):
            for n in range(len(word) - m + 1):
                stretch = tuple(word[n:n+m])
                word_prob += math.log(n_grams[m][stretch]) - math.log(n_grams[m-1][stretch[:-1]])

        return word_prob

    def top_words(self, length, n_top):
        def evaluator(n_grams, word, total_length, best_words):
            word = tuple(' ' + word)
            word_prob = self.get_word_prob(word)
            if not len(best_words):
                return True
            return word_prob > best_words[0][0]

        def visitor(n_grams, word, best_words):
            word = tuple(' ' + word + ' ')
            word_prob = self.get_word_prob(word)
            try_to_store(best_words, word, word_prob, MAX_WORDS_TO_STORE)

        def visit_words(prefix, alphabet, length, evaluate, visit):
            if length == 0:
                visit(prefix)
                return

            if evaluate(prefix):
                for a in alphabet:
                    visit_words(prefix + a, alphabet, length - 1, evaluate, visit)

        words = []

        best_words = []

        visit_words(
            u'',
            self.info.alphabet, length, lambda w : evaluator(self.n_grams, w, length, best_words),
            lambda w : visitor(self.n_grams, w, best_words)
        )

        for prob, word_tuple in list(reversed(sorted(best_words, key=itemgetter(0))))[:n_top]:
            words.append(u''.join(word_tuple[1:-1])) # trim spaces from ends

        return words


# returns true iff this word is good enough to go into the heap
def try_to_store(best_words, word, word_prob, max_to_store):
    if len(best_words) < max_to_store:
        heapq.heappush(best_words, (word_prob, word))
        return True

    (bad_prob, bad_word) = heapq.heappushpop(best_words, (word_prob, word))
    return bad_word != word
