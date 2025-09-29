import os
from typing import BinaryIO, Dict, Tuple
import regex as re
from concurrent.futures import ProcessPoolExecutor
import time
import pickle

def find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int,
    split_special_token: bytes,
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert isinstance(split_special_token, bytes), "Must represent special token as a bytestring"

    # Get total file size in bytes
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    # Initial guesses for chunk boundary locations, uniformly spaced
    # Chunks start on previous index, don't include last index
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

    for bi in range(1, len(chunk_boundaries) - 1):  # bi [1, desired_num_chunks - 1]
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)  # Start at boundary guess
        while True:
            # each time read mini_chunk_size, until read the split token or EOF, then set the chunk_bondary
            # the boundary only ensure end with split_special_token, but not ensure the number of split_special_token in each chunk.

            mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

            # If EOF, this boundary should be at the end of the file
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            # Find the special token in the mini chunk
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))

# This class is used as a dict value to indicate a byte pair's frequency and all the origin of the pair.
# data: freq indicates the frequency of the pair
#       idxs stores word index that include this pair
class BPE_pair():
    # used when a new pair is found and created
    def __init__(self, freq: int, word_index: int):
        self.freq = freq
        self.idxs = [word_index]
    
    # used when another word has the same pair, update freq and indexes
    def add_freq(self, freq: int, word_index: int):
        self.freq = self.freq + freq
        self.idxs.append(word_index)
    
    # used when merge a pair.
    # for example, when merge ('s','t'), word 'fast' should remove pair('a', 's') while word 'gas' does not do this remove
    def decrease_freq(self, freq: int, word_index: int):
        self.freq = self.freq - freq
        self.idxs.remove(word_index)

# call this function when merge pair in word, return the begin pos
def search_pair_in_word(word: tuple, pair: tuple) -> int:
    for pos in range(0, len(word)-1):
        if(word[pos] == pair[0] and word[pos+1] == pair[1]):
            return pos

# split chunk with special_tokens, and encode each token into utf-8
# return a generator for tuple[bytes, bytes....]
# like ('p', 'y', 't', 'h', 'o', 'n')
# special_tokens should be passed in training, but no need to pass in encode
# encoding should encode special_tokens on its own.
def parse_chunk(chunk, special_tokens = None, debug = None):
    if special_tokens is not None:
        escaped_tokens = [re.escape(token) for token in special_tokens]
        pattern = "|".join(escaped_tokens)
        docs = re.split(pattern, chunk)
    else: # handle encode condition that no special_tokens
        docs = [chunk]
    for doc in docs:
        words = re.finditer(PAT, doc)
        for word in words:
            word = word.group().encode('utf-8')
            word = tuple([bytes([token]) for token in word])
            yield word

# call this function when word tuple is updated
# for example when ('s', 't') is to be merged
# word ('f','a','s','t') should update to ('f','a','st')
def merge_tuple_elem(tup: tuple, pos: int)-> tuple:
    new_elem = tup[pos] + tup[pos+1]
    new_tup = tup[:pos] + (new_elem,) + tup[pos+2:]
    return new_tup

def dump_data_to_file(file_path, data):
    with open(file_path, "wb") as f:
        pickle.dump(data, f)

def load_data_from_file(file_path):
    with open(file_path, 'rb') as f:
        return pickle.load(f)

'''
data:
    paris: tuple -> BPE_pair dict, store all bpe pair
    index_to_word: int -> tuple dict, store map between index to word, word is stored into tuple instead of str to perform merge
    index_to_freq: int -> int, store map between index to freq, index is the same in index_to_word, so the word -> freq is stored
    next_index: int, the index for incoming word used in index_to_word
    merges: record the merge order of tokens
    vocab: int -> bytes, store all vocab
'''
class BPE_pairs_manager():
    def __init__(self, vocab):
        self.pairs: Dict[Tuple[str], BPE_pair] = {}
        self.index_to_word = {}
        self.index_to_freq = {}
        self.next_index = 0
        self.merges = []
        self.vocab = vocab

    # call this function when iterate all words after read file
    def parse_new_word(self, word: tuple, freq: int):
        index = self.next_index
        self.index_to_word[index] = word
        self.index_to_freq[index] = freq
        self.next_index = self.next_index + 1
        for pair in zip(word[:-1], word[1:]):
            self.add_pair(pair, freq, index)
    
    def add_pair(self, pair: tuple, freq: int, index: int):
        if pair in self.pairs:
            self.pairs[pair].add_freq(freq, index)
        else:
            self.pairs[pair] = BPE_pair(freq, index)

    def get_max_freq_pair(self):
        max_freq = 0
        max_pair_list = []
        for pair, pair_obj in self.pairs.items():
            if pair_obj.freq > max_freq:
                max_pair_list = [pair]
                max_freq = pair_obj.freq
            elif pair_obj.freq == max_freq:
                max_pair_list.append(pair)
        max_pair = max(max_pair_list)
        return max_pair

    '''
    what merge should do? assume ('e', 'a') is the chosen pair
    1. decrease related pair freq, for example, ('r', 'e', 'a', 'd'), decrease freq for ('r', 'e') and ('a', 'd')
    2. update related word tuple, ('r', 'e', 'a', 'd') -> ('r', 'ea', 'd')
    3. add new pair into self.pairs, ('r', 'ea', 'd') will introduce ('r', 'ea') and ('ea', d)
    all this 3 steps need the word index, fortunatly we can find the related indexes in BPE_pair.idxs
    4. remove ('s', 't') in self.pairs
    '''
    def merge(self):
        max_pair = self.get_max_freq_pair()
        max_pair_obj = self.pairs[max_pair]
        affected_idxs = max_pair_obj.idxs
        for index in affected_idxs:
            word = self.index_to_word[index]
            freq = self.index_to_freq[index]
            pos = search_pair_in_word(word, max_pair)
            # decrease related pair freq.
            if pos > 0:
                related_pair = (word[pos-1], word[pos])
                related_pair_obj = self.pairs[related_pair]
                related_pair_obj.decrease_freq(freq, index)
                # considering remove pair in pairs if related_pair_obj.freq is 0, but this should be a corner case I think
            if pos < len(word) - 2:
                related_pair = (word[pos+1], word[pos+2])
                related_pair_obj = self.pairs[related_pair]
                related_pair_obj.decrease_freq(freq, index)
            new_word = merge_tuple_elem(word, pos)
            self.index_to_word[index] = new_word
            if pos > 0:
                new_pair = (new_word[pos-1], new_word[pos])
                self.add_pair(new_pair, freq, index)
            if pos < len(new_word) - 1:
                new_pair = (new_word[pos], new_word[pos+1])
                self.add_pair(new_pair, freq, index)
        self.merges.append(max_pair)
        new_vocab_item = max_pair[0] + max_pair[1]
        new_vacab_index = len(self.vocab)
        self.vocab[new_vacab_index] = new_vocab_item
        _ = self.pairs.pop(max_pair, None)

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

## Usage
def train_bpe(input_path: str, vocab_size: int, special_tokens: list[str]):
    vocab = {}
    for i in range(256):
        vocab[i] = bytes([i])
    for i, special_token in enumerate(special_tokens):
        vocab[256 + i] = special_token.encode('utf-8')

    with open(input_path, "rb") as f:
        num_processes = 4
        boundaries = find_chunk_boundaries(f, num_processes, b"<|endoftext|>")

        # The following is a serial implementation, but you can parallelize this
        # by sending each start/end pair to a set of processes.
        word_freq: Dict[Tuple[str], int] = {}
        m = BPE_pairs_manager(vocab)
        for start, end in zip(boundaries[:-1], boundaries[1:]):
            f.seek(start)
            chunk = f.read(end - start).decode("utf-8", errors="ignore")
            # Run pre-tokenization on your chunk and store the counts for each pre-token
            docs = re.split(r"<\|endoftext\|>", chunk)
            for doc in docs:
                words = re.finditer(PAT, doc)
                for word in words:
                    word = word.group()
                    word = tuple(word)
                    if word in word_freq:
                        word_freq[word] = word_freq[word] + 1
                    else:
                        word_freq[word] = 1
        # print(word_freq)
        # ('file read and word freq count is over')
        for word, freq in word_freq.items():
            m.parse_new_word(word, freq)
        train_epochs = vocab_size - len(vocab)
        for _ in range(train_epochs):
            m.merge()
        print(m.index_to_word)
        out_file = open('../data/result.txt', "w", encoding='utf-8')
        for k, v in m.vocab.items():
            out_file.write(f"{k}: {v}\n")
        for k, v in m.index_to_word.items():
            out_file.write(f"{k}: {v}\n")
        return m.vocab, m.merges

# parallel read file to generate word_freq dict
def parallism_launch_function(filepath: str, chunk_boundaries: list[tuple], special_tokens: list[str]):
    word_freq: Dict[Tuple[str], int] = {}
    with open(filepath, "rb") as f:
        for start, end in chunk_boundaries:
            f.seek(start)
            # replace is used to avoid difference between Linux and windows, I perform this lab in win, so the raw data is corrupted.
            # chunk = f.read(end - start).replace(b'\r\n', b'\n').decode("utf-8", errors="ignore")
            chunk = f.read(end - start).decode("utf-8", errors="ignore")
            # Run pre-tokenization on your chunk and store the counts for each pre-token
            words = parse_chunk(chunk, special_tokens)
            for word in words:
                if word in word_freq:
                    word_freq[word] = word_freq[word] + 1
                else:
                    word_freq[word] = 1
        f.close()
    return word_freq

def train_bpe_with_parallism(input_path: str, vocab_size: int, special_tokens: list[str], debug = None) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    vocab = {}
    for i in range(256):
        vocab[i] = bytes([i])
    for i, special_token in enumerate(special_tokens):
        vocab[256 + i] = special_token.encode('utf-8')
    word_freq: Dict[Tuple[str], int] = {}
    m = BPE_pairs_manager(vocab)
    with open(input_path, "rb") as f:
        # My 136kf has 20 logic processor, so can try 20 processes, awsome!
        num_processes = 20
        chunk_per_process = 4
        boundaries = find_chunk_boundaries(f, num_processes * chunk_per_process, b"<|endoftext|>")
        f.close()
        chunk_num = len(boundaries) -1
        boundaries = [(start, end) for start, end in zip(boundaries[:-1], boundaries[1:])]

        # since chunk num maybe lower than expect, we should recalc exact processes, but this should be the same if dataset is bigggggg.
        boundaries_per_process = [boundaries[i:i+chunk_per_process] for i in range(0, chunk_num, chunk_per_process)]
        num_processes = len(boundaries_per_process)
        with ProcessPoolExecutor(max_workers=num_processes) as executor:
            futures = [
                executor.submit(parallism_launch_function, input_path, boundaries_per_process[i], special_tokens) for i in range(num_processes)
            ]
            for future in futures:
                executor_res = future.result()
                for word, freq in executor_res.items():
                    if word in word_freq:
                        word_freq[word] += freq
                    else:
                        word_freq[word] = freq

    # print('file read and word freq count is over')
    for word, freq in word_freq.items():
        m.parse_new_word(word, freq)
    train_epochs = vocab_size - len(vocab)
    for _ in range(train_epochs):
        m.merge()
    # print(m.index_to_word)
    if debug is not None:
        out_file = open('../data/result.txt', "w", encoding='utf-8')
        # out_file = open('result.txt', "w", encoding='utf-8')
        for k, v in enumerate(m.merges):
            out_file.write(f"{k}: {v}\n")
        for k, v in m.vocab.items():
            out_file.write(f"{k}: {v}\n")
        for k, v in m.index_to_word.items():
            out_file.write(f"{k}: {v}\n")
        out_file.close()
    return m.vocab, m.merges

class tokenizer():
    def __init__(self, vocab: dict[int, bytes], merges: list[tuple[bytes, bytes]], special_tokens: str = None):
        self.special_tokens = None
        self.token_to_idx: Dict[bytes, int] = {}
        self.special_token_to_idx: Dict[str, int] = {} # the given special_token is str
        if special_tokens is not None:
            # sort the input special_tokens so can pass duplicate tokens test
            special_tokens = sorted(special_tokens, key=len, reverse=True)
            self.special_tokens = special_tokens
            for special_token in special_tokens:
                special_token = special_token.encode('utf-8')
                if special_token not in vocab.values():
                    vocab_size = len(vocab)
                    vocab[vocab_size] = special_token
        
        for idx, token in vocab.items():
            self.token_to_idx[token] = idx
        self.idx_to_token = vocab
        # we need to encode the input text by the sequence of merges. This means we need to iterate the merges list
        # but we only need to iterate the merges once
        self.merges = merges
    
    def from_files(self, vocab_filepath:str, merges_filepath:str, special_tokens=None):
        vocab = load_data_from_file(vocab_filepath)
        merges = load_data_from_file(merges_filepath)
        self.__init__(vocab, merges, special_tokens)

    def encode(self, text: str) -> list[int]:
        encoded = []
        if self.special_tokens is not None:
            escaped_tokens = [re.escape(token) for token in self.special_tokens]
            pattern = "(" + "|".join(escaped_tokens) + ")"  # capature special_token as well
            docs = re.split(pattern, text)
        else:
            docs = [text]
        for i in range(len(docs)):
            if i%2 == 0:
                words = parse_chunk(docs[i])
                for word in words:
                    for merge in self.merges:
                        # a word may include several merge so use loop
                        while (pos := search_pair_in_word(word, merge)) is not None:
                            word = merge_tuple_elem(word, pos)
                    # turn bytes into vocab index
                    word = [self.token_to_idx[token] for token in word]
                    encoded = encoded + word
            else:   # special_token will only appear in odd index
                id = self.token_to_idx[docs[i].encode('utf-8')]
                encoded = encoded +[id]
        return encoded
    def encode_iterable(self, iter):
        while True:
            chunk = ''.join(iter.readline() for _ in range(100))
            if not chunk:
                break
            tokens = self.encode(chunk)
            for token in tokens:
                yield token

    def decode(self, ids: list[int]) -> str:
        decoded = b''
        for id in ids:
            if id in self.idx_to_token:
                word = self.idx_to_token[id]
                decoded = decoded + word
            else:
                decoded = decoded + b'\xff\xfd'
        return decoded.decode('utf-8', 'replace')

if __name__ == '__main__':
    start_time = time.time()
    train_data_set_path = '../data/TinyStoriesV2-GPT4-valid.txt'
    target_vocab_size = 1000
    vocab_special_tokens = ["<|endoftext|>"]
    #vocab, merges = train_bpe(train_data_set_path, target_vocab_size, vocab_special_tokens)
    vocab, merges = train_bpe_with_parallism(train_data_set_path, target_vocab_size, vocab_special_tokens, debug = True)
    vocab_path = '../data/vocab.bin'
    merges_path = '../data/merges.bin'
    dump_data_to_file(vocab_path, vocab)
    dump_data_to_file(merges_path, merges)

    vocab = load_data_from_file(vocab_path)
    merges = load_data_from_file(merges_path)
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"time used: {elapsed_time:.6f} seconds")
    #print(merges)
    T = tokenizer(vocab, merges, ["<|endoftext|>"])
    encoded = T.encode('The strange <|endoftext|>park<|endoftext|>')
    decoded = T.decode(encoded)
    print(encoded)
    print(decoded)

# uv run pytest tests/test_train_bpe.py
# uv run pytest tests/test_tokenizer.py