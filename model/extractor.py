import PyPDF2
import re

def clean_text(text):
    text = re.sub(r'https?://\\S+|www\\.\\S+', '', text)
    text = re.sub(r'[^a-za A-Z0-9 .,\\n]', '', text)
    text = re.sub(r'\\n+', ' ', text)
    text = re.sub(r'\\s+', ' ', text).strip()
    return text

def extract_text_from_pdf(pdf_path):
    text_blocks = []
    with open(pdf_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        for page in reader.pages:
            text = page.extract_text()
            if text:
                text = clean_text(text)
                blocks = [text[i:i+500] for i in range(0, len(text), 500)]
                text_blocks.extend(blocks)
    return text_blocks

import json

def extract_questions(json_path):
    with open(json_path, 'r') as file:
        questions_data = json.load(file)
        questions = {list(item.keys())[0]: list(item.values())[0] for item in questions_data.get("questions", [])}
    return questions

def export_answers(answers, output_path):
    with open(output_path, 'w') as output_file:
        formatted_answers = [{key: value} for key, value in answers.items()]
        json.dump(formatted_answers, output_file, indent=4, ensure_ascii=False)