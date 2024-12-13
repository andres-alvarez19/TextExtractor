from transformers import pipeline
from retrieval import find_relevant_block
from extractor import extract_questions

def ask_flan_t5(question, context):
    qa_pipeline = pipeline("text2text-generation", model="google/flan-t5-base")
    input_text = f"Contexto: {context}\nPregunta: {question}"
    response = qa_pipeline(input_text, max_length=100, truncation=True)
    return response[0]['generated_text']

def process_questions_from_json(json_path, text_blocks):
    answers = {}

    for question in extract_questions(json_path):
        relevant_block = find_relevant_block(question, text_blocks)
        answer = ask_flan_t5(question, relevant_block)
        answers[question] = answer

    return answers