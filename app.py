from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from profile_mapel import mapel_profiles
from google.oauth2.service_account import Credentials
from googleapiclient.discovery import build

SERVICE_ACCOUNT_FILE = 'service_acc.json'

SCOPES = ['https://www.googleapis.com/auth/spreadsheets']

credential = Credentials.from_service_account_file(SERVICE_ACCOUNT_FILE, scopes=SCOPES)

SPREADSHEET_ID = '14Qy5WOXJQMuA-yN7wC424e8K8C5NW_w3OYmiaOL5HLaA'

RANGE_NAME = 'Sheet1!A2'

app = FastAPI()

# Middleware CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Normalisasi nilai sesuai skala fitur
def normalize_profile(values):
    value_ranges = (
        [(0, 100)] * 5 + [(1, 5)] * 6 + [(0, 100)] * 6
    )
    return np.array([(v - mn) / (mx - mn) for v, (mn, mx) in zip(values, value_ranges)])




# Fungsi rekomendasi
def recommend_subjects(student_raw_profile, top_n=5):
    student_norm = normalize_profile(student_raw_profile).reshape(1, -1)
    mapel_names = list(mapel_profiles.keys())
    mapel_vectors = np.array([normalize_profile(profile) for profile in mapel_profiles.values()])
    similarities = cosine_similarity(student_norm, mapel_vectors)[0]
    df = pd.DataFrame({
        "Mata Pelajaran": mapel_names,
        "Skor Kecocokan": similarities
    }).sort_values(by="Skor Kecocokan", ascending=False).reset_index(drop=True)
    return df.head(top_n)

class StudentProfile(BaseModel):
    akademik: list[int]
    keminatan: list[int]
    riasec: list[int]

def save_to_sheet(student: StudentProfile, recommendations: list[dict]):
    try:
        service = build('sheets', 'v4', credentials=credential)
        sheet = service.spreadsheets()

        profile_data = student.akademik + student.keminatan + student.riasec
        recommendation_data = [rec['Mata Pelajaran'] for rec in recommendations]
        
        data_row = profile_data + recommendation_data
        values = [
            data_row
        ]
        body = {
            'values': values
        }
        result = sheet.values().append(
            spreadsheetId=SPREADSHEET_ID,
            range=RANGE_NAME,
            valueInputOption='RAW',
            insertDataOption='INSERT_ROWS',
            body=body
        ).execute()

        print(f"{result.get('updatedCells')} cells updated.")
    except Exception as e:
        print(f"An error occurred: {e}")
# Fungsi untuk menyimpan data ke Google Sheets

@app.get("/")
def read_root():
    return {"message": "Welcome to the Subject Recommendation API!"}

@app.post("/recommend/")
def recommend(student: StudentProfile, top_n: int = 5):
    student_raw_profile = student.akademik + student.keminatan + student.riasec
    recommendations = recommend_subjects(student_raw_profile, top_n)

    save_to_sheet(student, recommendations)

    return recommendations.to_dict(orient="records")
