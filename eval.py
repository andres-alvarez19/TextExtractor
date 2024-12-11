from evaluate import load
import torch
from transformers import AutoTokenizer, AutoModelForQuestionAnswering, TrainingArguments, Trainer
from datasets import Dataset
import json

# Load the model and tokenizer
model = AutoModelForQuestionAnswering.from_pretrained("./trained_model")
tokenizer = AutoTokenizer.from_pretrained("./trained_model")

# Load the dataset manually from JSON
with open("custom_squad.json", "r") as f:
    data = json.load(f)

# Prepare the data for `datasets`
def extract_data(data):
    examples = []
    for item in data["data"]:
        for paragraph in item["paragraphs"]:
            context = paragraph["context"]
            for qa in paragraph["qas"]:
                answer = qa["answers"][0]
                if answer["answer_start"] >= 0:  # Filter out invalid answers
                    examples.append({
                        "question": qa["question"],
                        "context": context,
                        "id": qa["id"],
                        "start_position": answer["answer_start"],
                        "end_position": answer["answer_start"] + len(answer["text"])
                    })
    return examples

flat_data = extract_data(data)

# Create a Hugging Face Dataset
dataset = Dataset.from_list(flat_data)
split_dataset = dataset.train_test_split(test_size=0.2)
tokenized_datasets = split_dataset.map(
    lambda x: tokenizer(
        x["question"],
        x["context"],
        truncation=True,
        padding="max_length",
        max_length=512
    ),
    batched=True
)

# Add labels to the tokenized dataset
def add_labels(example):
    inputs = tokenizer(example["context"], truncation=True, padding="max_length", max_length=512)
    start_positions = inputs.char_to_token(example["start_position"])
    end_positions = inputs.char_to_token(example["end_position"] - 1)

    if start_positions is None or start_positions >= tokenizer.model_max_length:
        start_positions = 0
    if end_positions is None or end_positions >= tokenizer.model_max_length:
        end_positions = 0

    example["start_positions"] = start_positions
    example["end_positions"] = end_positions
    return example

tokenized_datasets = tokenized_datasets.map(add_labels, batched=False)

# Configure training
training_args = TrainingArguments(
    output_dir="./results",
    learning_rate=2e-5,
    num_train_epochs=3,
    per_device_train_batch_size=16,
    save_steps=10_000,
    save_total_limit=2
)

# Custom Trainer class
class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        start_positions = inputs.pop("start_positions")
        end_positions = inputs.pop("end_positions")
        outputs = model(**inputs)
        if outputs.loss is None:
            start_logits, end_logits = outputs.start_logits, outputs.end_logits
            loss_fn = torch.nn.CrossEntropyLoss()
            start_loss = loss_fn(start_logits, start_positions)
            end_loss = loss_fn(end_logits, end_positions)
            loss = (start_loss + end_loss) / 2
        else:
            loss = outputs.loss
        return (loss, outputs) if return_outputs else loss

trainer = CustomTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["test"],
    processing_class=tokenizer,
)

# Load metrics
metric = load("squad")

def evaluate_qa_model(trainer, eval_dataset):
    results = []
    for example in eval_dataset:
        question = example["question"]
        context = example["context"]
        start_pos = example["start_positions"]
        end_pos = example["end_positions"]
        gold_answer = context[start_pos:end_pos]

        inputs = tokenizer(question, context, truncation=True, padding="max_length", max_length=512, return_tensors="pt")
        inputs = {key: value.to(trainer.model.device) for key, value in inputs.items()}
        with torch.no_grad():
            outputs = trainer.model(**inputs)
        start_logits = outputs.start_logits[0]
        end_logits = outputs.end_logits[0]

        start_idx = torch.argmax(start_logits)
        end_idx = torch.argmax(end_logits)
        predicted_answer = tokenizer.decode(inputs["input_ids"][0][start_idx:end_idx + 1])

        results.append({
            "id": example["id"],
            "gold_answer": gold_answer,
            "predicted_answer": predicted_answer
        })

        metric.add(
            prediction={"id": example["id"], "prediction_text": predicted_answer},
            reference={"id": example["id"], "answers": {"text": [gold_answer], "answer_start": [start_pos]}}
        )

    metrics = metric.compute()
    return metrics, results

# Evaluate the model
metrics, results = evaluate_qa_model(trainer, tokenized_datasets["test"])
print("Metrics:", metrics)

for res in results[:5]:
    print(f"Question ID: {res['id']}")
    print(f"Gold Answer: {res['gold_answer']}")
    print(f"Predicted Answer: {res['predicted_answer']}")
    print("-" * 50)