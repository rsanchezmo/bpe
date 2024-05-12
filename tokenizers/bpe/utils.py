from typing import List, Dict, Tuple


def get_consecutive_pairs(indices: List[int]) -> Dict[Tuple[int, int], int]:
    """Returns a dictionary of consecutive pairs of indices"""
    pairs = {}
    for left, right in zip(indices[:-1], indices[1:]):
        pairs[(left, right)] = pairs.get((left, right), 0) + 1
    return pairs


def encode_utf8(txt: str) -> List[int]:
    """Encodes a string to a list of UTF-8 bytes and returns it as a list of integers """
    return list(txt.encode('utf-8'))


if __name__ == '__main__':
    text = "hello ll hello"
    encoded = encode_utf8(text)
    pairs = get_consecutive_pairs(encoded)
    print(encoded)
    print(pairs)

