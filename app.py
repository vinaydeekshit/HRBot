from flask import Flask, render_template, request, jsonify
import pdfplumber
from sentence_transformers import SentenceTransformer, util
from langchain.llms import Ollama
import spacy

# Initialize models
app = Flask(__name__)

retrieval_model = SentenceTransformer("multi-qa-mpnet-base-dot-v1")  # Better for Q&A
ollama = Ollama(model="llama3.2:latest")  # Language model for response generation
nlp = spacy.load("en_core_web_sm")  # For sentence splitting

generic_keywords = {
    "hello", "hi", "hey", "how are you", "good morning", "good afternoon", "good evening",
    "hi there", "hey there", "hello there", "what's up", "howdy", "greetings", "what's happening",
    "how's it going", "how can I help you today", "what can I do for you", "nice to meet you",
    "good to see you", "what's new", "how are things", "how's your day", "how's everything",
    "long time no see", "what can I assist you with", "hello again", "thank you", "thanks", "thanks a lot", 
    "thank you very much", "thanks so much", "thanks a ton", "I appreciate it", "I'm grateful", "many thanks", 
    "much appreciated", "thanks for your help", "thanks for everything", "I can't thank you enough", 
    "thank you so much", "you've been a big help", "thanks a million", "I owe you one", "thank you kindly", 
    "I appreciate your assistance", "thank you for your support", "goodbye", "bye", "see you later", "take care", 
    "farewell", "see you soon", "catch you later", "have a great day", "bye bye", "talk to you later", 
    "see you around", "take it easy", "so long", "until next time", "have a good one", "peace out", "later", 
    "bye for now", "see you", "until we meet again"
}

def extract_text_from_pdf(pdf_path):
    extracted_text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            if text:
                extracted_text += text
    return extracted_text.strip()

def split_into_chunks(text, chunk_size=512):
    chunks = []
    current_chunk = []
    current_length = 0

    for sentence in nlp(text).sents:
        sentence_text = sentence.text.strip()
        if current_length + len(sentence_text) > chunk_size:
            if current_chunk:
                chunks.append(" ".join(current_chunk))
            current_chunk = [sentence_text]
            current_length = len(sentence_text)
        else:
            current_chunk.append(sentence_text)
            current_length += len(sentence_text)
    
    if current_chunk:
        chunks.append(" ".join(current_chunk))
    
    return chunks

def retrieve_relevant_context(chunks, query, top_k=3):
    chunk_embeddings = retrieval_model.encode(chunks, convert_to_tensor=True)
    query_embedding = retrieval_model.encode(query, convert_to_tensor=True)
    scores = util.pytorch_cos_sim(query_embedding, chunk_embeddings)[0]
    top_results = scores.topk(k=top_k)
    relevant_chunks = [chunks[idx] for idx in top_results.indices]
    return " ".join(relevant_chunks)

def generate_simple_response_with_llama(context, question):
    prompt = f"""
    Based on the following context, provide a clear and polite response to the question in 2-3 sentences.
    Avoid using special symbols, and ensure the response is concise, friendly, and without any inappropriate language.

    Context: {context}

    Question: {question}
    Answer:
    """
    response = ollama(prompt)
    return response.strip()

def is_generic_question(question):
    question_lower = question.strip().lower()
    return question_lower in generic_keywords or len(question.split()) < 3

@app.route('/index')
def home():
    return render_template('index.html')


@app.route('/')
def welcome():
    return render_template('welcome.html')

@app.route('/get_answer', methods=['POST'])
def process_pdf():
    # Access the question from the JSON body
    data = request.get_json()  # This parses the JSON data from the request
    question = data.get('question')  # Get the 'question' field

    # Check if the question exists
    if not question:
        return jsonify({'error': 'No question provided'}), 400

    # Define the path to the PDF file stored in the project directory
    pdf_path = r'combined_HR_POLICY.pdf'  # Modify this path as per your PDF's location
    
    # Check if the question is a generic one
    if is_generic_question(question):
        context = ""
    else:
        document_text = extract_text_from_pdf(pdf_path)
        chunks = split_into_chunks(document_text)
        context = retrieve_relevant_context(chunks, question)
    
    answer = generate_simple_response_with_llama(context, question)
    
    return jsonify({'answer': answer})


if __name__ == "__main__":
    app.run(debug=True)
