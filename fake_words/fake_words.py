# -*- coding: utf-8 -*-
import codecs
import itertools
import math
import heapq
import re
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
* (str) The plaintext dictionary for the language
* (str) The text corpus for the language
* (str) The pretty name of the text corpus for the language
* (unicode) The alphabet of the language, listing accented characters separately
"""
LanguageInfo = namedtuple("LanguageInfo", ["name", "dictionary_text", "corpus", "corpus_name", "alphabet"])


def _open_file_simple(file_path, encoding):
    return codecs.open(join(BASE_PATH, file_path), 'r', encoding=encoding)


def load_gutenberg_text(file_path):
    """
    Texts from Project Gutenberg have English-language copyright notices at the beginning and end. This method pulls out
    the good stuff from the middle.
    """
    gutenberg_regex = ".*START OF THIS PROJECT GUTENBERG EBOOK(.*)END OF THIS PROJECT GUTENBERG EBOOK.*"
    regex = re.compile(gutenberg_regex, flags=re.DOTALL)

    return regex.match(_open_file_simple(file_path, "utf-8").read()).group(0)


def _parse_ispell_file(lines):
    """
    ISpell files seem to be composed of lines, where each line is either blank, or composed of a word, whitespace,
    then a number.
    """
    regex = re.compile("^((.+)\s+(.+))?$")

    words = []
    for line in lines:
        empty, word, number = regex.match(line).groups()
        if empty is not None:
            words.append(word)

    return "\n".join(words)


def load_ispell_file(file_path, encoding):
    return _parse_ispell_file(_open_file_simple(file_path, encoding).readlines())


def load_derewo(file_path, encoding):
    # Strip off the header
    document = _open_file_simple(file_path, encoding).read()
    file_regex = re.compile(".*---\n(.*)", flags=re.DOTALL)

    document, = file_regex.match(document).groups()

    # Parse the rest of the file using ISpell style.
    return _parse_ispell_file(document.split("\n"))


def load_odm():
    document = _open_file_simple("sjp/odm.txt", "windows-1250").read()
    return "\n".join(re.findall(r"[\w'\-]+", document, flags=re.UNICODE))



LANGUAGES = {
    "english": LanguageInfo(
        "english",
        _open_file_simple("TWL06/index.txt", "utf-8").read(),
        load_gutenberg_text("tale/index.txt"),
        "A Tale of Two Cities",
        u"abcdefghijklmnopqrstuvwxyz'"
    ),
    "french": LanguageInfo(
        "french",
        _open_file_simple("ods5/index.txt", "utf-8").read(),
        load_gutenberg_text("lesmis/index.txt"),
        "Les Misérables",
        u"aàâäbcçdeéèêëfghiîïjklmnoöôpqrstuûüùvwxyz'"
    ),
    "german": LanguageInfo(
        "german",
        load_derewo("derewo/derewo-v-100000t-2009-04-30-0.1", "ISO-8859-15"),
        load_gutenberg_text("siddhartha/index.txt"),
        "Siddhartha",
        u"aäbcdefghijklmnoöpqrsßtuüvwxyz"
    ),
    "italian": LanguageInfo(
        "italian",
        _open_file_simple("zingarelli2005/index.txt", "utf-8").read(),
        load_gutenberg_text("viva/index.txt"),
        "La vita italiana durante la rivoluzione francese e l'Impero",
        u"aàbcdeèéfghiìlmnoòpqrstuùvz'"
    ),
    "latin": LanguageInfo(
        "latin",
        _open_file_simple("whitaker/index", "utf-8").read(),
        load_gutenberg_text("cicero/index.txt"),
        "Cicero's Orations",
        u"abcdefghilmnopqrstuvwxz"
    ),
    "polish": LanguageInfo(
        "polish",
        load_odm(),
        load_gutenberg_text("pan-tadeusz/index.txt"),
        "Pan Tadeusz",
        u"aąbcćdeęfghilłmnńoóprsśtuwyzźż"
    ),
    "spanish": LanguageInfo(
        "spanish",
        _open_file_simple("lemario_spanish_dict/index.txt", "iso-8859-1").read(),
        load_gutenberg_text("belarmino/index.txt"),
        "Belarmino y Apolonio",
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
        for word in self.info.dictionary_text.split():
            self.dictionary.add(word.lower())

        # Initialize word frequency model
        self.alphabet_with_space = self.info.alphabet + u" "

        # Train models, keeping track of strange things we find in the training data
        self.n_grams = {}
        self.found_punctuation = defaultdict(int)
        self.found_strange_letters = defaultdict(int)
        self._train_frequency_model(min_gram, max_gram)

    def _generate_n_grams(self, min_gram, max_gram):
        for k in range(min_gram, max_gram + 1):
            grams = {}
            for l in itertools.product(self.info.alphabet + " ", repeat=k):
                grams[l] = 1
            self.n_grams += [grams]

    def _train_frequency_model(self, min_gram, max_gram):
        # First, clean the document by replacing non-approved punctuation with a space, the start/end token
        document = list(self.info.corpus.lower())
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
                for i in range(len(word_padded) - k + 1):
                    self.n_grams[k][tuple(word_padded[i:i+k])] += 1

        # print u"{} punctuation: {}".format(self.info.name, u" ".join(punctuation))
        # print u"{} strange letters: {}".format(self.info.name, u" ".join(strange_letters))

    def sample_progressive(self, n_times):
        """
        Sample words by going a letter at a time, from left to right.

        :param int n_times: the number of samples to take
        :return list[str]: a list of words sampled
        """
        sampled = list()
        for i in xrange(n_times):
            # Start with a start token
            word_so_far = ' '

            while True:
                # Choose the next letter given the word so far
                next_probs = np.empty(len(self.alphabet_with_space))
                for j, next_letter in enumerate(self.alphabet_with_space):
                    next_probs[j] = self.get_word_log_likelihood(word_so_far + next_letter)
                next_probs = e ** next_probs
                next_letter = choice(list(self.alphabet_with_space), p=next_probs / next_probs.sum())
                word_so_far += next_letter

                # Stop the sample when we sample an end token
                if next_letter == ' ':
                    break

            # Remove the start and end tokens
            sampled.append(word_so_far[1:-1])

        return sampled

    def sample_mh(self, word_length, max_to_store, n_runs, n_samples_per_run):
        """
        A Metropolis-Hastings sampler for words of a given length.

        This stores a sample at the end of every run.

        :param int word_length: the length of the words to sample
        :param int max_to_store: how many highest-likelihood words we should store
        :param int n_runs: the number of times to restart (also the number of samples returned)
        :param int n_samples_per_run: the number of samples to take before run
        :return ([(str, float)], [(str, float)]): a list of samples w/ likelihoods, and a heap top-likelihood words w/
                                                  likelihoods
        """

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
            word_prob = self.get_word_log_likelihood(u" {} ".format("".join(word)))
            for j in xrange(n_samples_per_run):
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

                # Decide whether or not to accept the new sample
                new_prob = self.get_word_log_likelihood(u" {} ".format("".join(new_word)))
                if random() < e ** (new_prob - word_prob):
                    word, word_prob = new_word, new_prob

                    word_str = u"".join(word)

                    # Store the word in our top-likelihood list, if appropriate
                    if word_str not in [top_word for top_prob, top_word in top]:
                        try_to_store(top, word_str, word_prob, max_to_store)

            # Record this sample at the end of the run
            samples.append((word_str, word_prob))

        return samples, top

    def get_word_log_likelihood(self, word):
        """
        Note that this is not necessarily the actual log likelihood of the word, because this is the product of several
        probability distributions. It's off by a constant.

        :param str word: the word for which to calculate likelihood
        :return float: the log of a number that is proportional to the likelihood of this word
        """
        n_grams = self.n_grams

        word_prob = 0
        # Take the product of all n_gram models
        for m in range(1, len(n_grams)):

            # Look at each stretch of the word
            for n in range(len(word) - m + 1):
                stretch = tuple(word[n:n+m])
                # This line calculates the log likelihood of word[n+m-1] given word[n:n+m-1] using the m-gram model.
                # This is the same this as the probability of word[n+m-1] given word[:n+m-1], because of the Markov
                # property.
                word_prob += math.log(n_grams[m][stretch]) - math.log(n_grams[m-1][stretch[:-1]])

        return word_prob

    def top_words(self, length, n_top):
        """
        :param int length: the length of words to search
        :param int n_top: the number of pairs to return
        :return [(float, str)]: the n_top words with highest likelihoods, in a heap
        """
        def evaluator(n_grams, word, total_length, best_words):
            """
            Check if this word has a high enough likelihood to continue searching.
            """
            word = tuple(' ' + word)
            word_prob = self.get_word_log_likelihood(word)
            if not len(best_words):
                return True
            return word_prob > best_words[0][0]

        def visitor(n_grams, word, best_words):
            """
            Check if this word has a high likelihood and, if so, store it.
            """
            word = tuple(' ' + word + ' ')
            word_prob = self.get_word_log_likelihood(word)
            try_to_store(best_words, word, word_prob, MAX_WORDS_TO_STORE)

        def visit_words(prefix, alphabet, length, evaluate, visit):
            """
            Visit words in a DFS order, pruning when evaluate returns False.
            """
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


def try_to_store(best_words, word, word_prob, max_to_store):
    """
    Try to store this word in a heap, but only if the word's likelihood is high enough.
    """
    if len(best_words) < max_to_store:
        heapq.heappush(best_words, (word_prob, word))
        return True

    (bad_prob, bad_word) = heapq.heappushpop(best_words, (word_prob, word))
    return bad_word != word
