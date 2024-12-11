from transformers import AutoTokenizer, AutoModelForQuestionAnswering, TrainingArguments, Trainer
from datasets import Dataset, DatasetDict
import torch
import json

# Cargar el modelo y el tokenizador
tokenizer = AutoTokenizer.from_pretrained("bert-large-uncased-whole-word-masking-finetuned-squad")
model = AutoModelForQuestionAnswering.from_pretrained("bert-large-uncased-whole-word-masking-finetuned-squad")

# Cargar los datos desde JSON
with open("custom_squad.json", "r") as f:
    data = json.load(f)


# Preparar los datos
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

# Crear un Dataset de Hugging Face
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


# Agregar las etiquetas al dataset tokenizado
def add_labels(example):
    inputs = tokenizer(example["context"], truncation=True, padding="max_length", max_length=512, stride=128, return_overflowing_tokens=True)
    start_positions = inputs.char_to_token(example["start_position"])
    end_positions = inputs.char_to_token(example["end_position"] - 1)

    # Asegurar que las posiciones sean válidas
    if start_positions is None or start_positions >= tokenizer.model_max_length:
        start_positions = 0  # Asignar al token [CLS] o inicio del contexto
    if end_positions is None or end_positions >= tokenizer.model_max_length:
        end_positions = 0  # Asignar al token [CLS] o inicio del contexto

    example["start_positions"] = start_positions
    example["end_positions"] = end_positions
    return example

# Aplicar esta función al dataset
tokenized_datasets = tokenized_datasets.map(add_labels, batched=False)

# Configurar el entrenamiento
training_args = TrainingArguments(
    output_dir="./results",
    learning_rate=1e-5,
    per_device_train_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
    save_steps=500,
    save_total_limit=2,
    logging_dir="./logs",
    logging_steps=100,
    max_steps=-1,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    save_strategy="steps"
)


# Subclase personalizada de Trainer con cálculo manual de pérdida
class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        # Extraer las posiciones de inicio y fin de las etiquetas
        start_positions = inputs.pop("start_positions")
        end_positions = inputs.pop("end_positions")

        # Pasar los inputs al modelo
        outputs = model(**inputs)

        # Calcular la pérdida manualmente si outputs.loss es None
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

# Entrenar el modelo
trainer.train()

# Guardar el modelo y el tokenizador
trainer.save_model("./trained_model")
tokenizer.save_pretrained("./trained_model")
