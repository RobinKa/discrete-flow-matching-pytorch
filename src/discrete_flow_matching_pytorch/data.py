import os

import datasets


def load_tiny_stories(tokenizer, split):
    def tokenize_function(examples):
        return tokenizer(
            examples["text"], truncation=True, padding="max_length", max_length=128
        )

    # Load dataset
    def load_split(split):
        dataset = datasets.load_dataset("roneneldan/TinyStories", split=split)
        dataset = dataset.map(tokenize_function, batched=True, num_proc=os.cpu_count())
        dataset.set_format(type="torch", columns=["input_ids"])
        return dataset

    return load_split(split)
