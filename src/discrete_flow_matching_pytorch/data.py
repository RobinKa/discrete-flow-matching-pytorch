import os

import datasets
import torch
from jsonargparse import CLI
from more_itertools import chunked
from tqdm import tqdm
from transformers import AutoTokenizer


def get_default_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.add_special_tokens({"pad_token": "[PAD]", "mask_token": "[MASK]"})
    return tokenizer


def load_tiny_stories(tokenizer, split: str, max_length: int = 128):
    def tokenize_function(examples):
        return dict(
            input_ids=tokenizer(
                examples["text"],
                truncation=True,
                padding="max_length",
                max_length=max_length,
            )["input_ids"]
        )

    # Load dataset
    def load_split(split):
        dataset = datasets.load_dataset("roneneldan/TinyStories", split=split)
        dataset = dataset.map(tokenize_function, batched=True, num_proc=os.cpu_count())
        dataset.set_format(type="torch", columns=["input_ids"])
        return dataset

    return load_split(split)


def load_squad(
    tokenizer, split, max_length_question: int = 32, max_length_answer: int = 8
):
    def tokenize_function(examples):
        question_tokens = tokenizer(
            examples["question"],
            truncation=True,
            padding="max_length",
            max_length=max_length_question,
        )

        answer_tokens = tokenizer(
            [row["text"][0] for row in examples["answers"]],
            truncation=True,
            padding="max_length",
            max_length=max_length_answer,
        )

        question_tokens = torch.tensor(question_tokens["input_ids"], dtype=torch.long)
        answer_tokens = torch.tensor(answer_tokens["input_ids"], dtype=torch.long)

        input_ids = torch.cat([question_tokens, answer_tokens], dim=-1)
        should_noise = torch.cat(
            [
                torch.zeros_like(question_tokens, dtype=torch.bool),
                torch.ones_like(answer_tokens, dtype=torch.bool),
            ],
            dim=-1,
        )

        return dict(input_ids=input_ids, should_noise=should_noise)

    def load_split(split):
        dataset = datasets.load_dataset("rajpurkar/squad", split=split)
        dataset = dataset.map(tokenize_function, batched=True, num_proc=os.cpu_count())
        dataset.set_format(type="torch", columns=["input_ids", "should_noise"])
        return dataset

    return load_split(split)


def load_github_code(
    tokenizer,
    split: str,
    languages: list[str] | None,
    licenses: list[str] | None,
    max_length: int = 128,
):
    assert split in ["train", "validation"], split

    # github code does not have a validation split, but we can use different seeds
    if split == "validation":
        shuffle_seed = 0
        split = "train"
    else:
        shuffle_seed = 1

    def tokenize_function(examples):
        # List to store the chunks for all examples in a batch
        all_chunks = {"input_ids": []}

        # Tokenize each code example and split into chunks of max_length
        for code in examples["code"]:
            # Tokenize the entire code snippet without truncation
            tokens = tokenizer(code, truncation=False, padding=False)["input_ids"]

            # Split tokens into chunks of max_length
            for chunk in chunked(tokens, max_length):
                # Pad the chunk to max_length if needed
                if len(chunk) < max_length:
                    chunk += [tokenizer.pad_token_id] * (max_length - len(chunk))

                # Append each chunk to the list under "input_ids"
                all_chunks["input_ids"].append(chunk)

        return all_chunks

    # Load dataset
    def load_split(split):
        dataset: datasets.IterableDataset = datasets.load_dataset(
            "codeparrot/github-code",
            split=split,
            streaming=True,
            languages=languages,
            licenses=licenses,
            filter_languages=languages is not None,
            filter_licenses=licenses is not None,
        )
        dataset = dataset.select_columns(["code"])
        dataset = dataset.map(tokenize_function, batched=True, remove_columns=["code"])
        dataset = dataset.with_format(type="torch")
        dataset = dataset.shuffle(seed=shuffle_seed)
        return dataset

    return load_split(split)


def load_dataset_by_name(dataset: str, tokenizer, split: str):
    match dataset:
        case "squad":
            return load_squad(tokenizer, split)
        case "tiny_stories":
            return load_tiny_stories(tokenizer, split)
        case "github_code":
            return load_github_code(tokenizer, split, languages=None, licenses=None)
        case "github_code_dockerfile_mit":
            return load_github_code(
                tokenizer, split, languages=["Dockerfile"], licenses=["mit"]
            )
        case "github_code_python_mit":
            return load_github_code(
                tokenizer, split, languages=["Python"], licenses=["mit"]
            )
        case _:
            raise ValueError(f"Unknown dataset {dataset}")


def main(
    dataset: str = "squad",
    split: str = "train",
    rows_to_print: int = 1,
    benchmark: bool = False,
):
    tokenizer = get_default_tokenizer()
    dataset = load_dataset_by_name(dataset=dataset, tokenizer=tokenizer, split=split)
    for i, row in tqdm(enumerate(dataset)):
        should_print = i < rows_to_print
        if should_print:
            print(row)
        if not should_print and not benchmark:
            break


if __name__ == "__main__":
    CLI(main)
