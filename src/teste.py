from datasets import load_from_disk
from collections import Counter

ds=load_from_disk('data/processed')
print(Counter(ds['train']['label']))