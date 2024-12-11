import json

# Cargar el archivo SQuAD
with open("custom_squad.json", "r") as f:
    squad_data = json.load(f)

# Almacenar las líneas con problemas
issues = []

# Validar cada QA para detectar desajustes
for article_idx, article in enumerate(squad_data["data"]):
    for paragraph_idx, paragraph in enumerate(article["paragraphs"]):
        context = paragraph["context"]
        for qa_idx, qa in enumerate(paragraph["qas"]):
            for answer_idx, answer in enumerate(qa["answers"]):
                if answer["answer_start"] >= len(context) or answer["answer_start"] < 0:
                    issues.append({
                        "article_idx": article_idx + 1,
                        "paragraph_idx": paragraph_idx + 1,
                        "qa_idx": qa_idx + 1,
                        "answer_idx": answer_idx + 1,
                        "context_length": len(context),
                        "answer_start": answer["answer_start"],
                        "id": qa["id"],
                        "question": qa["question"],
                        "answer_text": answer["text"]
                    })

# Mostrar resultados
for issue in issues:
    print(f"Article {issue['article_idx']}, Paragraph {issue['paragraph_idx']}, "
          f"QA {issue['qa_idx']}, Answer {issue['answer_idx']}:\n"
          f"  Question ID: {issue['id']}\n"
          f"  Question: {issue['question']}\n"
          f"  Answer Text: {issue['answer_text']}\n"
          f"  Answer Start: {issue['answer_start']} (Context Length: {issue['context_length']})\n")
