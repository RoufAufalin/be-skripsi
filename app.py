from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from profile_mapel import mapel_profiles

app = FastAPI()

# 🔓 Middleware CORS
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

@app.get("/")
def read_root():
    return {"message": "Welcome to the Subject Recommendation API!"}

@app.post("/recommend/")
def recommend(student: StudentProfile, top_n: int = 5):
    student_raw_profile = student.akademik + student.keminatan + student.riasec
    recommendations = recommend_subjects(student_raw_profile, top_n)
    return recommendations.to_dict(orient="records")
