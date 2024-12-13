from sentence_transformers import SentenceTransformer, util

def find_relevant_block(question, text_blocks):
    model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
    question_embedding = model.encode(question, convert_to_tensor=True)
    block_embeddings = model.encode(text_blocks, convert_to_tensor=True)
    similarities = util.cos_sim(question_embedding, block_embeddings)
    most_relevant_index = similarities.argmax().item()
    return text_blocks[most_relevant_index]