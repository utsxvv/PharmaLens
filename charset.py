"""
charset.py — Defines the character vocabulary for the CRNN model.

Every unique character the model can read must be listed here.
CTC requires a special blank token at index 0 — do not remove it.
"""

# All characters the model can predict
# Covers: uppercase, lowercase, digits, hyphen, dot, slash
CHARS = (
    "-"                          # hyphen  (e.g. 1-0-1)
    "."                          # dot     (e.g. Tab.)
    "/"                          # slash   (e.g. Rx/)
    "0123456789"                 # digits
    "ABCDEFGHIJKLMNOPQRSTUVWXYZ" # uppercase
    "abcdefghijklmnopqrstuvwxyz" # lowercase
)

# CTC blank token — always index 0
BLANK = 0

# Build lookup tables
#   char → index  (used during encoding labels)
#   index → char  (used during decoding predictions)
CHAR_TO_IDX = {char: idx + 1 for idx, char in enumerate(CHARS)}
IDX_TO_CHAR = {idx + 1: char for idx, char in enumerate(CHARS)}

# Total vocabulary size including blank token
NUM_CLASSES = len(CHARS) + 1   # +1 for blank at index 0


def encode(text: str) -> list[int]:
    """
    Convert a text string into a list of integer indices.

    Example:
        encode("Amox") → [10, 40, 48, 57]

    Characters not in CHARS are silently skipped.
    """
    return [CHAR_TO_IDX[c] for c in text if c in CHAR_TO_IDX]


def decode(indices: list[int]) -> str:
    """
    Convert a list of integer indices back to a text string.
    Skips blank tokens (index 0) and removes consecutive duplicates
    (standard CTC decoding — called greedy decoding).

    Example:
        decode([10, 10, 0, 40, 48, 48, 57]) → "Amox"
    """
    result    = []
    prev_idx  = None

    for idx in indices:
        if idx == BLANK:
            prev_idx = None    # reset on blank
            continue
        if idx != prev_idx:    # skip consecutive duplicates
            char = IDX_TO_CHAR.get(idx, "")
            if char:
                result.append(char)
        prev_idx = idx

    return "".join(result)


if __name__ == "__main__":
    print(f"Total characters : {len(CHARS)}")
    print(f"Vocabulary size  : {NUM_CLASSES}  (includes blank token)")
    print(f"Characters       : {CHARS}")
    print(f"\nEncode test : encode('Amox')  → {encode('Amox')}")
    print(f"Decode test : decode([10,10,0,40,48,48,57]) → '{decode([10,10,0,40,48,48,57])}'")
