from datasets import load_from_disk

dset = load_from_disk("../data/covertype")
print(dset)
