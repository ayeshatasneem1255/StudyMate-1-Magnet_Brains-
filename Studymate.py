!pip install sentence-transformers pymupdf faiss-cpu gradio transformers bitsandbytes -q 

  

import fitz 

import faiss 

import torch 

from sentence_transformers import SentenceTransformer 

from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig 

import gradio as gr 

  

# ---------------------------- 

# Load models (run once) 

# ---------------------------- 

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")  # smaller, faster 

  

bnb_config = BitsAndBytesConfig(load_in_8bit=True) 

granite_model_name = "ibm-granite/granite-7b-instruct" 

granite_tokenizer = AutoTokenizer.from_pretrained(granite_model_name) 

granite_model = AutoModelForCausalLM.from_pretrained( 

    granite_model_name, 

    device_map="auto", 

    quantization_config=bnb_config, 

) 

device = "cuda" if torch.cuda.is_available() else "cpu" 

granite_model.to(device) 

  

# ---------------------------- 

# Global variables 

# ---------------------------- 

vector_store = None 

chat_history = [] 

pdf_text = "" 

  

# ---------------------------- 

# Helper functions 

# ---------------------------- 

def extract_text_from_pdf(path): 

    text = "" 

    for page in fitz.open(path): 

        text += page.get_text() 

    return text 

  

def chunk_text(text, chunk_size=500, overlap=100): 

    words = text.split() 

    chunks, i = [], 0 

    while i < len(words): 

        chunks.append(" ".join(words[i:i+chunk_size])) 

        i += chunk_size - overlap 

    return chunks 

  

def build_faiss(chunks): 

    global vector_store, pdf_text 

    pdf_text = chunks 

    embeddings = embedding_model.encode(chunks) 

    index = faiss.IndexFlatL2(embeddings.shape[1]) 

    index.add(embeddings) 

    vector_store = index 

  

def search_index(query, k=3): 

    query_emb = embedding_model.encode([query]) 

    D, I = vector_store.search(query_emb, k) 

    return " ".join([pdf_text[i] for i in I[0]]) 

  

def answer_question(question): 

    retrieved_text = search_index(question) 

    prompt = f"{retrieved_text}\nQuestion: {question}\nAnswer:" 

    tokens = granite_tokenizer(prompt, return_tensors="pt").to(device) 

    output = granite_model.generate(**tokens, max_new_tokens=256, temperature=0.2) 

    answer_text = granite_tokenizer.decode(output[0], skip_special_tokens=True) 

    return answer_text.split("Answer:")[-1].strip() 

  

def process_pdfs(pdf_files): 

    if not pdf_files: 

        return "⚠️ Upload at least one PDF." 

    full_text = "" 

    for pdf in pdf_files: 

        full_text += extract_text_from_pdf(pdf.name) + "\n\n" 

    build_faiss(chunk_text(full_text)) 

    return f"✅ Processed {len(pdf_files)} PDF(s). Ask questions now!" 

  

def chat_query(user_msg): 

    if not user_msg.strip(): 

        return chat_history 

    answer = answer_question(user_msg) 

    chat_history.append(("user", user_msg)) 

    chat_history.append(("Ayesha_chatbot", answer)) 

    return chat_history 

  

# ---------------------------- 

# Gradio interface 

# ---------------------------- 

with gr.Blocks() as interface: 

    gr.Markdown("## 📘 StudyMate – Multi-PDF AI Chat (IBM Granite 7B)") 

    with gr.Row(): 

        pdf_input = gr.File(label="Upload PDFs", file_count="multiple", file_types=[".pdf"]) 

        process_btn = gr.Button("Process Files") 

    process_status = gr.Textbox(label="Status", interactive=False) 

    chatbot = gr.Chatbot(height=400) 

    with gr.Row(): 

        msg = gr.Textbox(label="Ask a question", lines=2) 

        send_btn = gr.Button("Send") 

    process_btn.click(process_pdfs, inputs=pdf_input, outputs=process_status) 

    send_btn.click(chat_query, inputs=msg, outputs=chatbot).then(lambda: "", None, msg) 

  

interface.launch() 

 
