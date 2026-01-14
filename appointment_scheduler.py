
"""
AI-Powered Appointment Scheduler Assistant
OCR (Tesseract) + NLP (Groq LLaMA3)
"""

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, Dict
from groq import Groq
from PIL import Image
import pytesseract
import os
import json
from datetime import datetime, timedelta
import pytz
import io
import re
import sqlite3
import uuid


# -------------------- TESSERACT PATH (WINDOWS) --------------------
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

ENV = os.getenv("ENV", "dev")
# -------------------- APP INIT --------------------
app = FastAPI(title="AI Appointment Scheduler ",    docs_url="/docs" if ENV == "dev" else None,
    redoc_url=None,
    openapi_url="/openapi.json" if ENV == "dev" else None)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------- GROQ CLIENT --------------------
client = Groq(api_key=os.getenv("GROQ_API_KEY"))
MODEL = "llama-3.1-8b-instant"

# DB_NAME = "appointments.db"
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_NAME = os.path.join(BASE_DIR, "appointments.db")

@app.get("/")
async def root():
    return {
        "status": "running",
        "service": "AI Appointment Scheduler"
    }


def clean_ocr_text(text: str) -> str:
    """
    Cleans common OCR noise and normalizes text
    """
    text = text.lower()

    # Replace OCR symbol noise
    replacements = {
        "@": " at ",
        "&": " and ",
        "|": " ",
        "_": " ",
        "-": " ",
        "\n": " ",
        "\t": " "
    }

    for k, v in replacements.items():
        text = text.replace(k, v)

    # Fix common OCR typos
    typo_map = {
        "nxt": "next",
        "tmrw": "tomorrow",
        "tomorow": "tomorrow",
        "todai": "today",
        "frday": "friday",
        "monay": "monday",
        "wednsday": "wednesday",
        "thurday": "thursday",
        "saturdy": "saturday",
        "appt": "appointment",
        "dentst": "dentist",
        "doc": "doctor"
    }

    words = text.split()
    corrected_words = [typo_map.get(word, word) for word in words]

    # Normalize spacing
    cleaned = " ".join(corrected_words)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()

    # Capitalize first letter for readability
    return cleaned.capitalize()



def init_db():
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS appointments (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        booking_id TEXT,
        department TEXT,
        date TEXT,
        time TEXT,
        timezone TEXT,
        department_conf REAL,
        date_conf REAL,
        time_conf REAL,
        overall_conf REAL,
        created_at TEXT
    )
    """)

    conn.commit()
    conn.close()

# Initialize DB at startup
init_db()


# -------------------- SCHEMAS --------------------
class TextRequest(BaseModel):
    text: str

class OCRResponse(BaseModel):
    raw_text: str
    confidence: float

class EntityResponse(BaseModel):
    entities: Dict[str, str]
    entities_confidence: float

class NormalizedResponse(BaseModel):
    normalized: Dict[str, str]
    normalization_confidence: float

class AppointmentResponse(BaseModel):
    appointment: Optional[Dict[str, str]]
    status: str
    confidence: Optional[Dict[str, float]] = None
    message: Optional[str] = None

# -------------------- CONSTANTS --------------------
DEPARTMENT_MAPPING = {
    "dentist": "Dentistry",
    "dental": "Dentistry",
    "doctor": "General Medicine",
    "physician": "General Medicine",
    "cardio": "Cardiology",
    "heart": "Cardiology",
    "ortho": "Orthopedics",
    "bone": "Orthopedics",
    "eye": "Ophthalmology",
    "neuro": "Neurology",
    "skin": "Dermatology",
    "derma": "Dermatology",
}

# -------------------- HELPERS --------------------
def now_ist():
    return datetime.now(pytz.timezone("Asia/Kolkata"))

def clean_text(text: str) -> str:
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def extract_date_regex(text: str, base_date: datetime) -> Optional[str]:
    text = text.lower()

    # today
    if "today" in text:
        return base_date.strftime("%Y-%m-%d")

    # tomorrow
    if "tomorrow" in text:
        return (base_date + timedelta(days=1)).strftime("%Y-%m-%d")

    # day after tomorrow
    if "day after tomorrow" in text:
        return (base_date + timedelta(days=2)).strftime("%Y-%m-%d")

    # next / coming weekday
    weekdays = {
        "monday": 0, "tuesday": 1, "wednesday": 2,
        "thursday": 3, "friday": 4, "saturday": 5, "sunday": 6
    }

    for day, idx in weekdays.items():
        if f"next {day}" in text or f"coming {day}" in text:
            days_ahead = (idx - base_date.weekday() + 7) % 7
            days_ahead = 7 if days_ahead == 0 else days_ahead
            return (base_date + timedelta(days=days_ahead)).strftime("%Y-%m-%d")

    # explicit date: 15 jan / 15 january
    match = re.search(r'(\d{1,2})\s*(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)', text)
    if match:
        day = int(match.group(1))
        month = ["jan","feb","mar","apr","may","jun","jul","aug","sep","oct","nov","dec"].index(match.group(2)) + 1
        year = base_date.year
        return datetime(year, month, day).strftime("%Y-%m-%d")

    # ISO date
    match_iso = re.search(r'\d{4}-\d{2}-\d{2}', text)
    if match_iso:
        return match_iso.group()

    return None


def extract_time_regex(text: str) -> Optional[str]:
    text = text.lower()

    match = re.search(r'(\d{1,2})\s*(am|pm)', text)
    if match:
        hour = int(match.group(1))
        meridian = match.group(2)

        if meridian == "pm" and hour != 12:
            hour += 12
        if meridian == "am" and hour == 12:
            hour = 0

        return f"{hour:02d}:00"

    match_24 = re.search(r'(\d{1,2}):(\d{2})', text)
    if match_24:
        return f"{int(match_24.group(1)):02d}:{match_24.group(2)}"

    return None


def groq_chat(prompt: str) -> str:
    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )
    return response.choices[0].message.content.strip()

# -------------------- API ENDPOINTS --------------------
@app.post("/api/ocr", response_model=OCRResponse)
async def extract_text_from_image(file: UploadFile = File(...)):
    """
    Step 1: REAL OCR using Tesseract
    """
    try:
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes))

        extracted_text = pytesseract.image_to_string(image)
        raw_text = clean_ocr_text(extracted_text)

        if not raw_text:
            raise ValueError("No text detected")

        return OCRResponse(
            raw_text=raw_text,
            confidence=0.90
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"OCR failed: {str(e)}")

@app.post("/api/extract-entities", response_model=EntityResponse)
async def extract_entities(req: TextRequest):
    """
    Step 2: Entity extraction using Groq
    """
    try:
        prompt = f"""
Extract appointment entities from:
"{req.text}"

Today's date is {now_ist().strftime('%Y-%m-%d %A')}

Return ONLY JSON:
{{
  "date_phrase": "...",
  "time_phrase": "...",
  "department": "..."
}}
"""
        result = groq_chat(prompt)
        result = result.replace("```json", "").replace("```", "").strip()

        return EntityResponse(
            entities=json.loads(result),
            entities_confidence=0.85
        )

    except Exception as e:
        raise HTTPException(500, f"Entity extraction failed: {str(e)}")


@app.post("/api/normalize", response_model=NormalizedResponse)
async def normalize_datetime(req: EntityResponse):
    ent = req.entities
    base_date = now_ist()

    combined_text = f"{ent.get('date_phrase','')} {ent.get('time_phrase','')}"

    # 1️⃣ DATE REGEX (FIRST)
    date_val = extract_date_regex(combined_text, base_date)

    # 2️⃣ TIME REGEX (SECOND)
    time_val = extract_time_regex(combined_text)

    if date_val and time_val:
        return NormalizedResponse(
            normalized={
                "date": date_val,
                "time": time_val,
                "tz": "Asia/Kolkata"
            },
            normalization_confidence=0.95
        )

    # 3️⃣ LLM FALLBACK
    try:
        prompt = f"""
Normalize appointment time.

Text: "{combined_text}"
Today: {base_date.strftime('%Y-%m-%d %A')}
Timezone: Asia/Kolkata

Return ONLY JSON:
{{
  "date": "YYYY-MM-DD",
  "time": "HH:MM",
  "tz": "Asia/Kolkata"
}}
"""
        raw = groq_chat(prompt)
        start = raw.find("{")
        end = raw.rfind("}") + 1
        data = json.loads(raw[start:end])

        return NormalizedResponse(
            normalized=data,
            normalization_confidence=0.85
        )

    except:
        # 4️⃣ HARD FALLBACK
        return NormalizedResponse(
            normalized={
                "date": base_date.strftime("%Y-%m-%d"),
                "time": "10:00",
                "tz": "Asia/Kolkata"
            },
            normalization_confidence=0.4
        )




@app.post("/api/finalize", response_model=AppointmentResponse)
async def finalize_appointment(
    entities: EntityResponse,
    normalized: NormalizedResponse
):
    dept_raw = entities.entities.get("department", "").lower()

    if not dept_raw:
        return AppointmentResponse(
            appointment=None,
            status="needs_clarification",
            message="Department not detected"
        )

    # Department confidence
    department = None
    dept_conf = 0.6

    for key, value in DEPARTMENT_MAPPING.items():
        if key in dept_raw:
            department = value
            dept_conf = 0.95
            break

    if not department:
        department = dept_raw.title()
        dept_conf = 0.7

    # Date & time confidence
    date_conf = normalized.normalization_confidence
    time_conf = normalized.normalization_confidence

    overall_conf = round(
        (dept_conf + date_conf + time_conf) / 3, 2
    )

    booking_id = save_appointment(
    appointment={
        "department": department,
        "date": normalized.normalized["date"],
        "time": normalized.normalized["time"],
        "tz": "Asia/Kolkata"
    },
    confidence={
        "department_confidence": dept_conf,
        "date_confidence": date_conf,
        "time_confidence": time_conf,
        "overall_confidence": overall_conf
    }
)

    return AppointmentResponse(
    status="ok",
    appointment={
        "booking_id": booking_id,
        "department": department,
        "date": normalized.normalized["date"],
        "time": normalized.normalized["time"],
        "tz": "Asia/Kolkata"
    },
    confidence={
        "department_confidence": dept_conf,
        "date_confidence": date_conf,
        "time_confidence": time_conf,
        "overall_confidence": overall_conf
    }
)



@app.post("/api/schedule-image", response_model=AppointmentResponse)
async def schedule_from_image(file: UploadFile = File(...)):
    """
    Full pipeline: Image → OCR → NLP → Final JSON
    """
    ocr = await extract_text_from_image(file)
    ent = await extract_entities(TextRequest(text=ocr.raw_text))
    norm = await normalize_datetime(ent)
    return await finalize_appointment(ent, norm)

@app.post("/api/schedule-text", response_model=AppointmentResponse)
async def schedule_from_text(req: TextRequest):
    ent = await extract_entities(req)
    norm = await normalize_datetime(ent)
    return await finalize_appointment(ent, norm)

@app.get("/health")
async def health():
    return {"status": "healthy", "ocr": "tesseract", "llm": "groq"}


def save_appointment(appointment: dict, confidence: dict) -> str:
    booking_id = f"APT-{uuid.uuid4().hex[:8].upper()}"
    created_at = now_ist().strftime("%Y-%m-%d %H:%M:%S")

    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()

    cursor.execute("""
    INSERT INTO appointments (
        booking_id, department, date, time, timezone,
        department_conf, date_conf, time_conf, overall_conf,
        created_at
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        booking_id,
        appointment["department"],
        appointment["date"],
        appointment["time"],
        appointment["tz"],
        confidence["department_confidence"],
        confidence["date_confidence"],
        confidence["time_confidence"],
        confidence["overall_confidence"],
        created_at
    ))

    conn.commit()
    conn.close()

    return booking_id

def get_appointment_by_booking_id(booking_id: str):
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()

    cursor.execute("""
    SELECT booking_id, department, date, time, timezone,
           department_conf, date_conf, time_conf, overall_conf,
           created_at
    FROM appointments
    WHERE booking_id = ?
    """, (booking_id,))

    row = cursor.fetchone()
    conn.close()

    if not row:
        return None

    return {
        "booking_id": row[0],
        "department": row[1],
        "date": row[2],
        "time": row[3],
        "tz": row[4],
        "confidence": {
            "department_confidence": row[5],
            "date_confidence": row[6],
            "time_confidence": row[7],
            "overall_confidence": row[8]
        },
        "created_at": row[9]
    }

@app.get("/api/appointment/{booking_id}")
async def get_appointment(booking_id: str):
    result = get_appointment_by_booking_id(booking_id)

    if not result:
        raise HTTPException(
            status_code=404,
            detail="Appointment not found"
        )

    return {
        "status": "ok",
        "appointment": {
            "booking_id": result["booking_id"],
            "department": result["department"],
            "date": result["date"],
            "time": result["time"],
            "tz": result["tz"],
            "created_at": result["created_at"]
        },
        "confidence": result["confidence"]
    }



# -------------------- RUN --------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)


