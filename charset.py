CHARS = (
    "-"                          
    "."                          
    "/"                          
    "0123456789"                 
    "ABCDEFGHIJKLMNOPQRSTUVWXYZ" 
    "abcdefghijklmnopqrstuvwxyz" 
)

BLANK = 0

CHAR_TO_IDX = {char: idx + 1 for idx, char in enumerate(CHARS)}
IDX_TO_CHAR = {idx + 1: char for idx, char in enumerate(CHARS)}

NUM_CLASSES = len(CHARS) + 1


def encode(text: str) -> list[int]:
    return [CHAR_TO_IDX[c] for c in text if c in CHAR_TO_IDX]


def decode(indices: list[int]) -> str:
    result    = []
    prev_idx  = None

    for idx in indices:
        if idx == BLANK:
            prev_idx = None    
            continue
        if idx != prev_idx:   
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
