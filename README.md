# Hallucination Detection and Reduction in Indian Legal AI

## Overview

Hallucinations in Large Language Models (LLMs) can generate factually incorrect or fabricated legal information, making them unreliable for high-stakes domains such as law. This project presents a comprehensive framework for detecting and reducing hallucinations in Indian legal AI systems.

The project consists of two complementary modules:

- **Hallucination Detection:** Evaluates hallucination behavior in InLegalBERT using embedding-based semantic similarity tests across multiple legal scenarios.
- **Hallucination Reduction:** Implements **Nyaya AI**, a Retrieval-Augmented Generation (RAG) based legal assistant that combines Groq Llama 3.3 with IndianKanoon to generate grounded, citation-supported legal responses.

Together, these modules provide an end-to-end approach for evaluating and improving the reliability of AI-powered legal assistants.

---

## Project Structure

```
Hallucination-Detection-and-Reduction-in-Indian-Legal-AI/
│
├── README.md
│
├── Hallucination_Detection/
│   ├── inlegalbert_demo.py
│   └── hallucination_results.json
│
└── Hallucination_Reduction/
    ├── app.py
    ├── legal_engine.py
    ├── metrics.py
    ├── index.html
    └── nyaya_metrics_20260401_091404.json
```

---

## Hallucination Detection

The detection module evaluates the reliability of **InLegalBERT** by measuring semantic similarity across carefully designed legal test cases. It identifies common hallucination patterns that may occur when language models are used without external knowledge retrieval.

### Features

- Section Number Hallucination Detection
- Negation Blindness Detection
- Jurisdiction Confusion Analysis
- Old Law vs. New Law Comparison
- Out-of-Distribution Detection
- Civil vs. Criminal Law Confusion
- Cosine Similarity Evaluation
- JSON Result Generation

---

## Hallucination Reduction (Nyaya AI)

The reduction module implements **Nyaya AI**, a Retrieval-Augmented Generation (RAG) based legal assistant that retrieves relevant legal information before generating responses. This approach significantly reduces hallucinations and improves factual accuracy.

### Features

- AI-powered Legal Question Answering
- Retrieval-Augmented Generation (RAG)
- IndianKanoon Case Retrieval
- Groq Llama 3.3 Integration
- Legal Document Analysis
- Citation-Supported Responses
- Hallucination Evaluation Metrics
- Interactive Web Interface

---

## Technologies Used

### Programming Language

- Python

### AI Models

- InLegalBERT
- Groq Llama 3.3 70B

### Frameworks

- FastAPI

### Frontend

- HTML
- CSS
- JavaScript

### APIs

- IndianKanoon API
- Groq API

### Libraries

- Hugging Face Transformers
- PyTorch
- Scikit-learn
- Requests

---

## Project Workflow

1. Detect hallucination patterns using InLegalBERT.
2. Analyze semantic inconsistencies across legal scenarios.
3. Retrieve relevant legal precedents from IndianKanoon.
4. Generate grounded legal responses using Groq Llama 3.3.
5. Evaluate hallucination reduction using dynamic performance metrics.

---

## Key Features

- Hallucination Detection
- Hallucination Reduction
- Retrieval-Augmented Generation (RAG)
- Legal Question Answering
- IndianKanoon Integration
- Legal Document Analysis
- Citation-Based Responses
- Dynamic Evaluation Metrics
- FastAPI Backend
- Interactive Web Interface

---

## Future Enhancements

- Multilingual legal support
- Improved citation accuracy
- Advanced legal document summarization
- Cloud deployment
- User authentication
- Support for additional legal datasets

---
