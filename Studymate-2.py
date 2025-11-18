# ============================ 

# 🚀 StudyMate – PDF Q&A System 

# IBM Granite 3.3-2B Instruct + FAISS + Gradio 

# ============================ 

  

!pip install transformers accelerate sentence-transformers faiss-cpu pymupdf gradio -q 

  

import fitz                           # PyMuPDF 

import faiss                          # Vector search 

import torch 

from transformers import AutoTokenizer, AutoModelForCausalLM 

from sentence_transformers import SentenceTransformer 

import gradio as gr 

  

# ---------------------------------------- 

# 1️⃣ Load Embedding Model & LLM (Granite) 

# ---------------------------------------- 

  

embed_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2") 

  

model_name = "ibm-granite/granite-3.3-2b-instruct" 

  

tokenizer = AutoTokenizer.from_pretrained(model_name) 

llm = AutoModelForCausalLM.from_pretrained( 

    model_name, 

    torch_dtype=torch.float16, 

    device_map="auto" 

) 

  

# ----------------------------- 

# 2️⃣ PDF → Text → Chunking 

# ----------------------------- 

def extract_text_from_pdf(pdf_path): 

    doc = fitz.open(pdf_path) 

    text = "" 

    for page in doc: 

        text += page.get_text() 

    return text 

  

  

def chunk_text(text, chunk_size=400): 

    words = text.split() 

    chunks = [] 

    for i in range(0, len(words), chunk_size): 

        chunk = " ".join(words[i:i+chunk_size]) 

        chunks.append(chunk) 

    return chunks 

  

  

# ------------------------------------- 

# 3️⃣ Create FAISS Vector DB 

# ------------------------------------- 

faiss_index = None 

text_chunks = [] 

  

def build_faiss(chunks): 

    global faiss_index, text_chunks 

  

    text_chunks = chunks 

    embeddings = embed_model.encode(chunks) 

  

    dim = embeddings.shape[1] 

    faiss_index = faiss.IndexFlatL2(dim) 

    faiss_index.add(embeddings) 

  

  

# ------------------------------------- 

# 4️⃣ Semantic Search + LLM Answer 

# ------------------------------------- 

def answer_question(query): 

    if faiss_index is None: 

        return "⚠️ Please upload PDF first." 

  

    # Convert question → embedding 

    q_emb = embed_model.encode([query]) 

  

    # Search FAISS 

    D, I = faiss_index.search(q_emb, k=3) 

    retrieved = "\n".join([text_chunks[i] for i in I[0]]) 

  

    # Prompt to LLM 

    prompt = f""" 

You are an academic assistant. Answer the question using ONLY the provided context. 

  

Context: 

{retrieved} 

  

Question: 

{query} 

  

Answer: 

""" 

  

    inputs = tokenizer(prompt, return_tensors="pt").to("cuda") 

    output = llm.generate(**inputs, max_new_tokens=300) 

    answer = tokenizer.decode(output[0], skip_special_tokens=True) 

  

    return answer 

  

  

# ------------------------------------- 

# 5️⃣ Gradio UI 

# ------------------------------------- 

def process_pdf(pdf_file): 

    if pdf_file is None: 

        return "Upload a PDF." 

  

    text = extract_text_from_pdf(pdf_file.name) 

    chunks = chunk_text(text) 

    build_faiss(chunks) 

  

    return "✅ PDF processed successfully! You can ask questions now." 

  

  

with gr.Blocks() as interface: 

    gr.Markdown("## 📘 StudyMate – AI-Powered PDF Q&A (IBM Granite 3.3-2B)") 

     

    pdf_input = gr.File(label="Upload PDF") 

    process_btn = gr.Button("Process PDF") 

    process_output = gr.Textbox(label="Status") 

  

    question = gr.Textbox(label="Ask a Question") 

    answer = gr.Textbox(label="Answer") 

  

    process_btn.click(process_pdf, inputs=pdf_input, outputs=process_output) 

    question.submit(answer_question, inputs=question, outputs=answer) 

  

interface.launch() 

 
