from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import pdfplumber
import json

# Cargar modelo y tokenizador de Flan-T5
model_name = "google/flan-t5-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

# Función para extraer texto del PDF
def extract_text_from_pdf(pdf_path):
    text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text += page.extract_text()
    return text

# Función para usar Flan-T5 en preguntas específicas
def ask_flan_t5(context, question):
    prompt = f"Context: {context}\n\nQuestion: {question}\nAnswer:"
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
    outputs = model.generate(inputs["input_ids"], max_length=128, num_beams=4, early_stopping=True)
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return answer

# Cargar PDF y dividirlo en fragmentos
pdf_path = "./sample_data/document2.pdf"  # Cambia por tu archivo
text = extract_text_from_pdf(pdf_path)
chunks = [text[i:i+2000] for i in range(0, len(text), 2000)]  # Fragmentos de 2000 caracteres

# Preguntas específicas
questions_with_keywords = {
    "What is the name of the animal or group of animals?": ["animal", "cow", "cattle", "pig", "chicken"],
    "How many animals are in the group?": ["number of animals", "herd size", "group size"],
    "How much manure was used in the study?": ["manure", "waste"],
    "What is the name of the antibiotic or antibiotics used in the animals?": ["antibiotic", "ceftiofur", "tetracycline"],
    "What is the name of the bacteria studied?": ["bacteria", "E. coli", "microorganism"],
    "To which antibiotics or antibiotic families was resistance detected?": ["resistance", "antibiotic", "resistance profile"]
}


import re

# Función para buscar fragmentos relevantes según palabras clave
def filter_relevant_chunks(chunks, keywords):
    relevant_chunks = []
    for chunk in chunks:
        if any(keyword.lower() in chunk.lower() for keyword in keywords):
            relevant_chunks.append(chunk)
    return relevant_chunks

# Procesar preguntas con fragmentos relevantes
responses = []
for question, keywords in questions_with_keywords.items():
    relevant_chunks = filter_relevant_chunks(chunks, keywords)
    for chunk in relevant_chunks:
        answer = ask_flan_t5(chunk, question)
        responses.append({"question": question, "context": chunk, "answer": answer})


# Guardar respuestas en un archivo JSON
output_path = "./sample_data/answers2.json"
with open(output_path, "w") as f:
    json.dump(responses, f, indent=4)

print(f"Respuestas guardadas en {output_path}")
