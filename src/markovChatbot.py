import random
import re
from typing import List, Tuple

from scipy.sparse import coo_matrix
from sklearn.preprocessing import normalize


class MarkovChatbot:
    """
    'naive' chatbot based on Markov chains
    """

    def __init__(self, corpus: str, n: int = 3):
        corpus = self._preprocess(corpus)
        self.corpus = [x for x in corpus.split(' ') if len(x) > 0]
        self.corpus_ids, self.ids_corpus = self._tokenize(self.corpus)
        self.n = n
        self.n_grams = self._create_ngrams()
        self.ngrams_ids, self.ids_ngrams = self._tokenize(self.n_grams)
        self.matrix = self._create_transition_matrix_proba()

    @staticmethod
    def _preprocess(line: str) -> str:
        line = re.sub(r'[^\w\s]', '', line)
        return line.lower()

    def _create_ngrams(self) -> List[str]:
        sequences = [self.corpus[i:] for i in range(self.n)]
        ngrams = [' '.join(ngram) for ngram in list(zip(*sequences))]
        return ngrams

    @staticmethod
    def _tokenize(text: List[str]) -> Tuple[dict, dict]:
        tokens = {}
        translation = {}
        text = list(set(text))
        for i, word in enumerate(text):
            tokens[word] = i
            translation[i] = word
        return tokens, translation

    def _create_transition_matrix_proba(self) -> coo_matrix:
        row, col, values = [], [], []

        for i in range(len(self.corpus[:-self.n])):
            ngram = ' '.join(self.corpus[i:i + self.n])
            next_word = self.corpus[i + self.n]
            ngram_id = self.ngrams_ids[ngram]
            word_id = self.corpus_ids[next_word]

            row.extend([ngram_id])
            col.extend([word_id])
            values.extend([1])

        matrix = coo_matrix((values, (row, col)), shape=(len(self.ngrams_ids), len(self.corpus_ids)))
        return normalize(matrix, norm='l1')

    def _random_ngram(self) -> str:
        ngram = random.choice(list(self.ngrams_ids.values()))
        return self.ids_ngrams[ngram]

    def _find_similar(self, sequence: List[str]) -> str:
        best_ngram, common = None, 0
        for ngram in self.ngrams_ids.keys():
            c = 0
            ngram_list = ngram.split(" ")
            for word in sequence:
                if word in ngram_list:
                    c += 1
            if c > common:
                best_ngram, common = ngram, c
        return best_ngram

    def _check_ngram(self, sequence: List[str]) -> str:
        if len(sequence) < self.n:
            ngram = self._find_similar(sequence)
        else:
            ngram = sequence[-self.n:]
            ngram = " ".join(ngram)
            if ngram not in self.ngrams_ids.keys():
                ngram = self._find_similar(sequence)
        return ngram

    def _generate_next_word(self, sequence: List[str]) -> str:
        ngram = self._check_ngram(sequence)
        ngram_id = self.ngrams_ids[ngram]
        proba = self.matrix[ngram_id].toarray()[0]

        word_id = random.choices(range(len(proba)), weights=proba, k=1)[0]
        return self.ids_corpus[word_id]

    def generate_response(self, inp: str, length: int) -> str:
        """
        generates response to user's input message
        :param inp: user's input message
        :param length: length of output message
        :return: message, response to user's input
        """
        sequence = self._preprocess(inp)
        sequence = [x for x in sequence.split(' ') if len(x) > 0]
        response = []
        response.extend(sequence)
        while len(response) != length + len(sequence):
            word = self._generate_next_word(response)
            response.append(word)
        response = response[len(sequence):]
        return " ".join(response)

    def fine_tune(self, new_corpus: str) -> None:
        """
        adds new corpus to model, tries to give it greater probability than base corpus
        :param new_corpus: corpus we want to add to our model
        """
        new_corpus = self._preprocess(new_corpus)
        new_corpus = [x for x in new_corpus.split(' ') if len(x) > 0]
        repeats = len(self.corpus) // len(new_corpus) + 5

        to_add = []
        for i in range(repeats):
            to_add.extend(new_corpus)

        self.corpus.extend(to_add)

        self.corpus_ids, self.ids_corpus = self._tokenize(self.corpus)
        self.n_grams = self._create_ngrams()
        self.ngrams_ids, self.ids_ngrams = self._tokenize(self.n_grams)
        self.matrix = self._create_transition_matrix_proba()


def chat_with_me(model: MarkovChatbot, steps: int = 5, len_message: int = 15):
    """
    enables chatting with MarkovChatbot
    :param model: chatbot based on Markov chains
    :param steps: number of iterations we wish to talk
    :param len_message: length of the model's response
    """
    for step in range(steps):
        inp = input(">>User:")
        response = model.generate_response(inp, len_message)
        print("Bot: {}".format(response))
