from flask import Flask, render_template, request, jsonify
import pdfplumber
from sentence_transformers import SentenceTransformer, util
from langchain.llms import Ollama
import spacy
import logging
import traceback

# Initialize the Flask app
app = Flask(__name__)

# Initialize models
retrieval_model = SentenceTransformer("multi-qa-mpnet-base-dot-v1")  # Retrieval model for Q&A
ollama = Ollama(model="llama3.2:latest")  # Language model for response generation
nlp = spacy.load("en_core_web_sm")  # For sentence splitting

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

# Generic keywords for simple/generic questions
generic_keywords = {
    "hello", "hi", "hey", "how are you", "good morning", "good afternoon", "good evening",
    "hi there", "hey there", "hello there", "what's up", "howdy", "greetings", "what's happening",
    "how's it going", "how can I help you today", "what can I do for you", "nice to meet you",
    "good to see you", "what's new", "how are things", "how's your day", "how's everything",
    "thank you", "thanks", "bye", "goodbye", "see you later", "take care"
}

# Helper Functions
def extract_text_from_pdf(pdf_path):
    """Extracts text from a PDF file."""
    try:
        extracted_text = ""
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                text = page.extract_text()
                if text:
                    extracted_text += text
        return extracted_text.strip()
    except Exception as e:
        logging.error(f"Error extracting text from PDF: {e}")
        raise

def split_into_chunks(text, chunk_size=512):
    """Splits text into smaller chunks for processing."""
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
    """Retrieves the most relevant context from text chunks."""
    try:
        chunk_embeddings = retrieval_model.encode(chunks, convert_to_tensor=True)
        query_embedding = retrieval_model.encode(query, convert_to_tensor=True)
        scores = util.pytorch_cos_sim(query_embedding, chunk_embeddings)[0]
        top_results = scores.topk(k=top_k)
        relevant_chunks = [chunks[idx] for idx in top_results.indices]
        return " ".join(relevant_chunks)
    except Exception as e:
        logging.error(f"Error retrieving relevant context: {e}")
        raise

def generate_simple_response_with_llama(context, question):
    """Generates a response using the Llama model."""
    try:
        prompt = f"""
        Based on the following context, provide a clear and polite response to the question in 2-3 sentences.
        Avoid using special symbols, and ensure the response is concise, friendly, and without any inappropriate language.

        Context: {context}

        Question: {question}
        Answer:
        """
        response = ollama(prompt)
        return response.strip()
    except Exception as e:
        logging.error(f"Error during Ollama call: {e}")
        raise

def is_generic_question(question):
    """Checks if the question is generic."""
    question_lower = question.strip().lower()
    return question_lower in generic_keywords or len(question.split()) < 3

# Flask Routes
@app.route('/')
def welcome():
    return render_template('welcome.html')

@app.route('/index')
def home():
    return render_template('index.html')

@app.route('/get_answer', methods=['POST'])
def process_pdf():
    """Processes the question and returns an answer."""
    try:
        # Parse JSON request
        data = request.get_json()
        question = data.get('question')
        if not question:
            return jsonify({'error': 'No question provided'}), 400

        # Define the path to the PDF file
        pdf_path = r'combined_HR_POLICY.pdf'

        # Check if the question is generic
        if is_generic_question(question):
            context = ""
        else:
            document_text = extract_text_from_pdf(pdf_path)
            chunks = split_into_chunks(document_text)
            context = retrieve_relevant_context(chunks, question)

        # Generate the answer
        answer = generate_simple_response_with_llama(context, question)
        return jsonify({'answer': answer})

    except Exception as e:
        logging.error(f"An error occurred: {e}")
        logging.error(traceback.format_exc())
        return jsonify({'error': 'Internal Server Error', 'details': str(e)}), 500

@app.route('/ollama-test', methods=['GET'])
def ollama_test():
    """Tests the Ollama service."""
    try:
        test_response = ollama("Hello, how are you?")
        return jsonify({'test_response': test_response.strip()})
    except Exception as e:
        logging.error(f"Ollama test failed: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/test-pdf', methods=['GET'])
def test_pdf():
    """Tests PDF extraction."""
    try:
        pdf_path = r'combined_HR_POLICY.pdf'
        text = extract_text_from_pdf(pdf_path)
        return jsonify({'pdf_text_preview': text[:500]})  # Return first 500 characters
    except Exception as e:
        logging.error(f"PDF extraction error: {e}")
        return jsonify({'error': str(e)}), 500

# Main Execution
if __name__ == "__main__":
    app.run(debug=True)
