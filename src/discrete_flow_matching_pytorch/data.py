import os

import datasets
import torch
import typer
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


def load_dataset_by_name(dataset: str, tokenizer, split: str):
    match dataset:
        case "squad":
            return load_squad(tokenizer, split)
        case "tiny_stories":
            return load_tiny_stories(tokenizer, split)
        case _:
            raise ValueError(f"Unknown dataset {dataset}")


def main(dataset: str = "squad", split: str = "train"):
    tokenizer = get_default_tokenizer()
    dataset = load_dataset_by_name(dataset=dataset, tokenizer=tokenizer, split=split)
    print(dataset[0])


if __name__ == "__main__":
    typer.run(main)
