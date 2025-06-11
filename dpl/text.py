import torch


def _word_tokenize(text: str, lower=True, filter_stopwords=False, filter_punctuation=False, user_filter=None):
    import nltk

    def __download_nltk():
        nltk.download('punkt_tab')
        nltk.download('punkt')
        nltk.download('stopwords')

    def __word_tokenize(_text, _lower, _filter_stopwords, _filter_punctuation, _user_filter):
        filter_set = set() if _user_filter is None else _user_filter
        if _filter_stopwords:
            stopwords = set(nltk.corpus.stopwords.words('english'))
            filter_set |= stopwords
        if _filter_punctuation:
            import string
            punctuation = set(string.punctuation)
            filter_set |= punctuation
        if _lower:
            _text = _text.lower()

        _text = nltk.word_tokenize(_text)  # 主要的性能消耗
        if len(filter_set) != 0:
            _text = [token for token in _text if token not in filter_set]

        return _text

    # 初次使用先运行 download_nltk
    try:
        return __word_tokenize(text, lower, filter_stopwords, filter_punctuation, user_filter)
    except LookupError:
        __download_nltk()
        return __word_tokenize(text, lower, filter_stopwords, filter_punctuation, user_filter)


class Vocab:
    def __init__(self, corpus: list[str], min_freq=0, max_vocab_size=None):
        self.token2idx = self.__build_vocab(corpus, min_freq, max_vocab_size)
        self.idx2token = list(self.token2idx.keys())
        self.vocab_size = len(self.idx2token)

    def decode(self, indices) -> list[str]:
        return [self.idx2token[idx] if idx < self.vocab_size else '<unk>' for idx in indices]

    def decode2str(self, indices) -> str:
        return ' '.join(self.decode(indices))

    def encode(self, tokens) -> torch.Tensor:
        return torch.tensor([self.token2idx[token]
                             if token in self.token2idx else self.vocab_size - 1 for token in tokens],
                            dtype=torch.int64)

    def encode_from_str(self, text: str) -> torch.Tensor:
        return self.encode(_word_tokenize(text))

    def __len__(self):
        return self.vocab_size

    @staticmethod
    def __build_vocab(corpus, min_frq, max_vocab_size):
        import collections

        token_counter = collections.Counter(corpus)
        token_counter.pop('<unk>', None)
        sorted_counter = sorted(token_counter.items(), key=lambda x: x[1])

        i = 0
        for token, count in sorted_counter:
            if count >= min_frq:
                break
            i += 1

        vocab_size_no_unk = len(sorted_counter) - i
        if max_vocab_size is not None and vocab_size_no_unk >= max_vocab_size:
            vocab_size_no_unk = max_vocab_size - 1
            i = len(sorted_counter) - vocab_size_no_unk

        vocab = {token: i for i, (token, _) in enumerate(sorted_counter[i:])}
        vocab['<unk>'] = vocab_size_no_unk
        return vocab


class Corpus:
    tensor: torch.Tensor

    def __init__(self, root: str, lower=True, filter_stopwords=False, filter_punctuation=False,
                 user_filter: set[str] = None):
        with open(root, 'r') as f:
            self.corpus = _word_tokenize(f.read(), lower, filter_stopwords, filter_punctuation, user_filter)

    def __len__(self):
        return len(self.corpus)

    def __getitem__(self, item):
        return self.corpus[item]

    def build_vocab(self, min_freq=0, max_vocab_size=None) -> Vocab:
        vocab = Vocab(self.corpus, min_freq, max_vocab_size)
        self.tensor = vocab.encode(self.corpus)
        print("vocab size: ", len(vocab))
        return vocab

    def get_tensor(self):
        return self.tensor
