# CS336 Spring 2025 Assignment 1: Basics

For a full description of the assignment, see the assignment handout at
[cs336_spring2025_assignment1_basics.pdf](./cs336_spring2025_assignment1_basics.pdf)

If you see any issues with the assignment handout or code, please feel free to
raise a GitHub issue or open a pull request with a fix.

## Setup

### Environment
We manage our environments with `uv` to ensure reproducibility, portability, and ease of use.
Install `uv` [here](https://github.com/astral-sh/uv) (recommended), or run `pip install uv`/`brew install uv`.
We recommend reading a bit about managing projects in `uv` [here](https://docs.astral.sh/uv/guides/projects/#managing-dependencies) (you will not regret it!).

You can now run any code in the repo using
```sh
uv run <python_file_path>
```
and the environment will be automatically solved and activated when necessary.

### Run unit tests


```sh
uv run pytest
```

Initially, all tests should fail with `NotImplementedError`s.
To connect your implementation to the tests, complete the
functions in [./tests/adapters.py](./tests/adapters.py).

### Download data
Download the TinyStories data and a subsample of OpenWebText

``` sh
mkdir -p data
cd data

wget https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-train.txt
wget https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-valid.txt

wget https://huggingface.co/datasets/stanford-cs336/owt-sample/resolve/main/owt_train.txt.gz
gunzip owt_train.txt.gz
wget https://huggingface.co/datasets/stanford-cs336/owt-sample/resolve/main/owt_valid.txt.gz
gunzip owt_valid.txt.gz

cd ..
```

## Personal words...
First of all, I finish all these part in Windows as I have set environment like pyTorch in Win, I don't want to set environment in VM again... 
Most part is smooth, but please attention that memory limit is not available in Win, and another problem is \n \r\n cause token freq diff, this may result a test fail, but I am confident my code is OK....  
Although the assignment suggusts to use Einsum, I just use pyTorch with matmul and bmm finish all part with a lot of personal annotation, hope this will be helpful.  

BPE training, encoding and decoding is in \cs336_basics\pretokenization_example.py  
Transformer architecture like RMS norm, transformer block etc is in \cs336_basics\transformer_architecture.py  
Training and sequence generating is in \cs336_basics\transformer_training.py  

You may see some hard coded path in my code, to be honsest I know that's bad, but I'm lazy sometime so please update it if needed.  

To run test, adapters.py in path /tests is changed, I import each part independently. And at the begining you can see  
``` python
# for import module in cs336_basics
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
```
this will locate to root of this project, and will cause a problem, you can see in transformer_training.py I import transformer_architecture and tokenizer, this will conflict with the root locate as the relative path has been changed. A stupid way to run related tests is change import as in project root use  

``` python
from cs336_basics/pretokenization_example import tokenizer
```
to replace
``` python
from pretokenization_example import tokenizer
```
temporarily. There should be an elegant way to handle this, but im not good in py and lazy, just let it go...