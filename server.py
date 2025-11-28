# --- SERVER AI CHO CICIDS 2018 ---

import uvicorn
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
import joblib
import pandas as pd
import numpy as np
from typing import List, Dict, Any
import os
import requests

LOGSTASH_RETURN = "http://192.168.2.24:5055/"


# --- CẤU HÌNH ---
app = FastAPI(
    title="CICIDS 2018 Anomaly Detection API",
    version="2.1.0"
)

# Biến toàn cục để chứa "dụng cụ"
artifacts = {}


@app.on_event("startup")
def load_artifacts():
    """Tải model và metadata khi server khởi động"""
    print("--- ĐANG KHỞI ĐỘNG SERVER AI (CICIDS 2018) ---")
    try:
        artifacts["model"] = joblib.load("model_cicids.joblib")
        artifacts["scaler"] = joblib.load("scaler_cicids.joblib")
        artifacts["meta"] = joblib.load("metadata_cicids.joblib")

        # Kiểm tra OHE (nếu có)
        if os.path.exists("ohe_cicids.joblib"):
            artifacts["ohe"] = joblib.load("ohe_cicids.joblib")
            print("-> Đã tải OneHotEncoder.")
        else:
            artifacts["ohe"] = None

        print("-> Đã tải Model, Scaler và Metadata thành công!")
        print(f"-> Ngưỡng cắt (Threshold): {artifacts['meta']['optimal_threshold']}")

    except Exception as e:
        print(f"LỖI: Không thể tải file joblib. {e}")
        print("Hãy chắc chắn các file .joblib nằm cùng thư mục server.py")


# --- INPUT SCHEMA CHUẨN ---
class LogRequest(BaseModel):
    logs: List[Dict[str, Any]]


class PredictionOutput(BaseModel):
    anomaly_score: float
    is_anomaly: bool
    threshold_used: float


# --- HÀM TIỀN XỬ LÝ ---
def preprocess_data(raw_data: List[Dict[str, Any]]) -> pd.DataFrame:
    df = pd.DataFrame(raw_data)
    meta = artifacts["meta"]

    df.columns = df.columns.str.strip()

    for col in meta['numerical_cols']:
        if col not in df.columns:
            df[col] = 0

    df = df[meta['numerical_cols']].copy()

    if 'constant_cols' in meta:
        df.drop(columns=meta['constant_cols'], errors='ignore', inplace=True)

    df = df.apply(pd.to_numeric, errors='coerce')
    df.replace([np.inf, -np.inf], 0, inplace=True)
    df.fillna(0, inplace=True)

    skewed_cols = [c for c in meta['skewed_cols'] if c in df.columns]
    if skewed_cols:
        df[skewed_cols] = df[skewed_cols].clip(lower=0)
        df[skewed_cols] = df[skewed_cols].apply(np.log1p)

    try:
        df_scaled = artifacts["scaler"].transform(df)
        df_final = pd.DataFrame(df_scaled, columns=df.columns, index=df.index)
    except Exception as e:
        print(f"Scaling error: {e}")
        raise HTTPException(status_code=500, detail=f"Preprocessing Error: {e}")

    return df_final


# --- API ENDPOINT NHẬN LOG TỪ LOGSTASH ---
# @app.post("/predict", response_model=List[PredictionOutput])
# async def predict(request: Request):
#     """
#     Nhận 2 dạng log:
#     1) Logstash gửi 1 event → {...}
#     2) Logstash gửi batch → {"logs": [ {...}, {...} ]}
#     """

#     body = await request.json()

#     # --- AUTO FIX INPUT FORMAT ---
#     # Logstash gửi từng log → không có "logs"
#     if isinstance(body, dict) and "logs" not in body:
#         logs = [body]                      # Gói thành mảng
#     else:
#         logs = body.get("logs", [])

#     if not logs:
#         raise HTTPException(status_code=400, detail="No logs received")

#     # 1. Preprocess
#     df_processed = preprocess_data(logs)

#     # 2. Predict
#     raw_scores = artifacts["model"].decision_function(df_processed)
#     anomaly_scores = -raw_scores

#     threshold = artifacts["meta"]['optimal_threshold']

#     results = []
#     for score in anomaly_scores:
#         results.append(PredictionOutput(
#             anomaly_score=float(score),
#             is_anomaly=bool(score > threshold),
#             threshold_used=threshold
#         ))

#     return results
@app.post("/predict", response_model=List[PredictionOutput])
async def predict(request: Request):

    body = await request.json()
    print("\n========== NHẬN YÊU CẦU TỪ LOGSTASH ==========")
    print("📥 Raw body nhận từ Logstash:")
    print(body)

    if isinstance(body, dict) and "logs" not in body:
        logs = [body]            # Logstash gửi 1 event → gói thành mảng
        print("📌 Logstash gửi 1 log. Đã chuyển thành logs[]")
    else:
        logs = body.get("logs", [])

    print("\n📥 Logs sau khi auto-fix:")
    print(logs)

    if not logs:
        print("❌ Không nhận được logs nào!")
        raise HTTPException(status_code=400, detail="No logs received")

    # --- 2) PREPROCESS ---
    print("\n🔧 Đang tiền xử lý dữ liệu...")
    df_processed = preprocess_data(logs)

    print("\n📊 DataFrame sau tiền xử lý:")
    print(df_processed)

    # --- 3) PREDICT ---
    raw_scores = artifacts["model"].decision_function(df_processed)
    anomaly_scores = -raw_scores
    threshold = artifacts["meta"]['optimal_threshold']

    print("\n⚙️ Kết quả anomaly score:")
    for score in anomaly_scores:
        print(f" - score = {score} (threshold = {threshold})")

    # --- 4) Trả về kết quả ---
    results = []
    for score in anomaly_scores:
        results.append(PredictionOutput(
            anomaly_score=float(score),
            is_anomaly=bool(score > threshold),
            threshold_used=threshold
        ))




    print("\n✅ Trả về kết quả cho Logstash:", results)
    print("=============================================\n")
    
    # try:
    #     requests.post(LOGSTASH_RETURN, json={"ai_results": [r.model_dump() for r in results]})
    #     print("Đã gửi log phân tích về Logstash")
    # except Exception as e:    
    #     print("LỖI gửi ngược về Logstash:", e)

    try:
        payload = {"ai_results": results}
        requests.post(LOGSTASH_RETURN, json=payload)
        print("Đã gửi log phân tích về Logstash")
    except Exception as e:
        print("LỖI gửi ngược về Logstash:", e)

    
    return results


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
