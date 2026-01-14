# AI-Powered Appointment Scheduler Assistant

## Overview
This project implements a backend service that converts **natural language text** or **image-based appointment requests** into structured appointment data.

It uses:
- **OCR (Tesseract)** for extracting text from images  
- **LLM-based NLP (Groq LLaMA 3)** for entity extraction and normalization  

The system is designed as a **multi-stage pipeline** with validation guardrails to handle noisy or ambiguous inputs.

---

## Problem Statement Chosen
**Problem Statement 1: AI-Powered Appointment Scheduler Assistant**

**Focus Area:**  
OCR → Entity Extraction → Normalization → Structured Output

---

## Architecture

Client (Text / Image)
|
v
+--------------------+
| OCR (Tesseract) | ← image only
+--------------------+
|
v
+--------------------+
| Entity Extraction | ← Groq LLaMA 3
+--------------------+
|
v
+--------------------+
| Normalization |
| (Asia/Kolkata) |
| Regex + LLM |
+--------------------+
|
v
+--------------------+
| Guardrails & |
| Final JSON Output |
+--------------------+
|
v
SQLite Database


---

## Tech Stack

- **FastAPI** – Backend framework  
- **Tesseract OCR** – Image text extraction  
- **Groq API (LLaMA 3)** – NLP entity extraction & normalization  
- **SQLite** – Local persistence  
- **Python 3.10+**

---

## Features

- Accepts **typed text** or **images (scanned notes / emails)**
- Handles **OCR noise** (e.g., `nxt`, `@`, spelling mistakes)
- Extracts:
  - Department
  - Date phrase
  - Time phrase
- Normalizes date & time to **Asia/Kolkata timezone**
- Implements **guardrails**:
  - Ambiguous date/time detection
  - Missing department detection
- Stores confirmed appointments with confidence scores
- REST API with Swagger documentation

---

## API Endpoints

### 1. OCR – Image to Text
**POST** `/api/ocr`

**Input:** Image file  

**Response:**
```json
{
  "raw_text": "Book dentist next Friday at 3pm",
  "confidence": 0.9
}
2. Entity Extraction

POST /api/extract-entities

Request:

{
  "text": "Book dentist next Friday at 3pm"
}
Response:

{
  "entities": {
    "date_phrase": "next Friday",
    "time_phrase": "3pm",
    "department": "dentist"
  },
  "entities_confidence": 0.85
}

3. Normalization (Asia/Kolkata)

POST /api/normalize

Response:

{
  "normalized": {
    "date": "2025-09-26",
    "time": "15:00",
    "tz": "Asia/Kolkata"
  },
  "normalization_confidence": 0.95
}

4. Final Appointment Creation

POST /api/schedule-text

Request:

{
  "text": "Book dentist next Friday at 3pm"
}


Success Response:

{
  "status": "ok",
  "appointment": {
    "booking_id": "APT-8F3A9C2D",
    "department": "Dentistry",
    "date": "2025-09-26",
    "time": "15:00",
    "tz": "Asia/Kolkata"
  }
}


Guardrail Response (Ambiguity):

{
  "status": "needs_clarification",
  "message": "Ambiguous date/time or department"
}

Sample cURL Requests
Text Input
curl -X POST http://127.0.0.1:8000/api/schedule-text \
-H "Content-Type: application/json" \
-d '{"text":"book dentist next friday at 3pm"}'

Image Input
curl -X POST http://127.0.0.1:8000/api/schedule-image \
-F "file=@note.jpg"

Setup Instructions
1. Install Dependencies
pip install fastapi uvicorn pytesseract pillow groq pytz

2. Install Tesseract OCR

Windows: https://github.com/tesseract-ocr/tesseract

Update Tesseract path in code if required

3. Set Environment Variable
export GROQ_API_KEY=your_api_key_here


Windows PowerShell:

$env:GROQ_API_KEY="your_api_key_here"

4. Run the Server
uvicorn appointment_scheduler:app


Open Swagger UI:

http://127.0.0.1:8000/docs
