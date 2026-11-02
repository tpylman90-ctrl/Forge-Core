"""Reversible UTF-8 byte tokenizer; token IDs are 0..255."""


class ByteTokenizer:
    vocab_size = 256

    def encode(self, text: str) -> list[int]:
        return list(text.encode("utf-8"))

    def decode(self, tokens: list[int]) -> str:
        if any(not isinstance(t, int) or t < 0 or t >= self.vocab_size for t in tokens):
            raise ValueError("Byte token IDs must be integers between 0 and 255")
        return bytes(tokens).decode("utf-8")
