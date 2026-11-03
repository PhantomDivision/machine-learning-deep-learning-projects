class SimpleTokenizer:

    def __init__(self, text):

        # Create vocabulary
        words = sorted(set(text.split()))

        self.word_to_id = {
            word: idx
            for idx, word in enumerate(words)
        }

        self.id_to_word = {
            idx: word
            for word, idx in self.word_to_id.items()
        }

    def encode(self, text):

        words = text.split()

        return [
            self.word_to_id[word]
            for word in words
        ]

    def decode(self, token_ids):

        return " ".join(
            self.id_to_word[token_id]
            for token_id in token_ids
        )