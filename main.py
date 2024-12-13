from model.extractor import extract_text_from_pdf, export_answers
from model.model import process_questions_from_json

if __name__ == "__main__":

    pdf_path = "./papers/document.pdf"
    text_blocks = extract_text_from_pdf(pdf_path)

    questions_path = "./questions/questions.json"
    qa = process_questions_from_json(questions_path, text_blocks)

    answers_path = "./answers/answers.json"
    export_answers(qa, answers_path)