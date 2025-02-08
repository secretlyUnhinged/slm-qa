from fastapi import FastAPI, UploadFile, File, Form
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import torch
import PyPDF2
import io

app = FastAPI()

# Load the tokenizer and model
model_name = "facebook/bart-large-cnn"  # Change if needed
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

# Store uploaded book text
book_text = ""

@app.post("/upload_book")
async def upload_book(file: UploadFile = File(...)):
    global book_text
    if file.filename.endswith(".pdf"):
        # Read PDF file
        pdf_reader = PyPDF2.PdfReader(io.BytesIO(await file.read()))
        book_text = "\n".join([page.extract_text() for page in pdf_reader.pages if page.extract_text()])
    else:
        # Read text file
        book_text = (await file.read()).decode("utf-8")
    
    return {"message": "Book uploaded successfully!", "character_count": len(book_text)}

@app.post("/ask_question")
async def ask_question(question: str = Form(...)):
    global book_text
    if not book_text:
        return {"error": "No book uploaded. Please upload a book first."}
    
    # Prepare input text (truncate to avoid exceeding model limit)
    context = book_text[:1000]  # Use first 1000 characters (adjust if needed)
    input_text = f"Question: {question} Context: {context}"

    # Tokenize input
    inputs = tokenizer(
        input_text, return_tensors="pt", truncation=False, max_length=1024
    )

    # Generate response with extended max tokens
    output = model.generate(
        inputs["input_ids"], 
        max_new_tokens=512,  # Increased response length
        temperature=0.7,
        top_p=0.9
    )

    # Decode output
    answer = tokenizer.decode(output[0], skip_special_tokens=True)

    return {"question": question, "answer": answer}

@app.get("/")
def home():
    return {"message": "Welcome to the SLM Q&A API! Upload a book and ask questions."}
