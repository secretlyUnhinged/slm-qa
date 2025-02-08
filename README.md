Approach-

The system follows a retrieval-augmented generation (RAG) approach by first identifying relevant text chunks and then generating answers using a sequence-to-sequence model.

Key Steps:

Text Extraction: Extracts text from uploaded PDF or text files.

Text Preprocessing & Chunking: Cleans the text and splits it into smaller, overlapping chunks.

Chunk Selection: Identifies the most relevant chunks for a given question using keyword-based similarity scoring.

Question Answering: Uses a fine-tuned transformer model to generate an answer based on the retrieved context.


Model Architecture-

The system consists of two primary components:

a) Chunk-based Retrieval

Text Chunking: The book is split into overlapping chunks (default: 1000 characters with 200-character overlap).

Keyword Matching: Identifies relevant chunks based on common words between the question and text.

b) Question Answering Model

Model Used: facebook/bart-large-cnn, a transformer-based summarization model fine-tuned for generating concise answers.

Tokenization: Utilizes the AutoTokenizer from Hugging Face for text processing.

Generation Parameters:

max_new_tokens=256: Controls response length.

num_beams=4: Improves diversity in answer generation.

temperature=0.7 and top_p=0.9: Fine-tune response randomness.

no_repeat_ngram_size=3: Avoids repeated phrases.



Preprocessing Techniques-

a) Text Cleaning

Removes extra spaces, fixes period spacing, and resolves formatting inconsistencies from PDFs.

Splits concatenated words caused by PDF extraction issues.

b) Text Chunking

Large documents are split into overlapping chunks for improved context.

Chunks are stored in memory for efficient retrieval.

c) Device Optimization

The system selects the best available device (CUDA, MPS, or CPU) for model inference.

Moves input tensors to the chosen device for efficiency.


Evaluation Methodology-

To assess the performance and reliability of the system, the following evaluation metrics are used:

a) Retrieval Evaluation (Chunk Selection Accuracy)

Metric: Measures whether the retrieved text contains the correct answer.

Method: Manually verifies if the selected chunks are relevant.

Improvement Strategy: Enhance keyword-based retrieval with semantic search models (e.g., FAISS + sentence transformers).

b) Question Answering Model Evaluation

Metrics:

Exact Match (EM): Measures whether the generated answer is identical to the ground truth.

F1-Score: Assesses the overlap between the generated and actual answer.

Baseline Comparison: Compares with simpler extractive models like tinyroberta-squad2 or DistilBERT.

Sample input/output:
"question": "what is scrum?",
  "answer": "Scrum is an agile method that focuses on managing iterative development rather than specific agile practices. The Scrum sprint cycle is fixed length, normally 2–4weeks long. The project closure phase wraps up the project, completes required documentation such assystem help frames and user manuals.",
  "context_length": 2001

Step 1: Set Up an Anaconda Environment

conda create --name slm_env python=3.9 -y
conda activate slm_env

Step 2: Install Dependencies

pip install transformers sentence-transformers faiss-cpu fastapi uvicorn pypdf pdfplumber

Step 3: Run the API

Save the script as slm_qa.py and run:

python slm_qa.py

This starts a FastAPI server on http://localhost:8000.

Step 4: Upload a Book

Step 5: Ask a Question

Note: if it does not give any output run this on terminal: uvicorn slm_q:app --host 0.0.0.0 --port 8080
and try opening: http://127.0.0.1:8080/docs

Observations:
Handling Long Texts for NLP Models
Full books are too large to process at once, so chunking with overlap is essential for preserving context.

Context Selection Matters
Using keyword-based retrieval helps find relevant text but may not be as precise as embedding-based retrieval (like FAISS).

The system generates abstractive answers using facebook/bart-large-cnn, which is effective for summarization but may not always extract precise answers.
Alternative models (like extractive QA models such as tinyroberta-squad2) could provide more direct answer extraction from retrieved text.
