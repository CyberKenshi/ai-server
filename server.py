
import uvicorn                
from fastapi import FastAPI   
from pydantic import BaseModel 
import joblib                 
import pandas as pd           
import numpy as np            
from typing import List, Dict, Any


app = FastAPI(
    title="Anomaly Detection API",
    description="Một API để phát hiện bất thường trong log mạng, tích hợp với ELK Stack.",
    version="1.0.0"
)

artifacts = {
    "model": None,
    "scaler": None,
    "ohe": None,
    "le": None,
    "lists": None
}

@app.on_event("startup")
def load_artifacts():
    """
    Hàm này sẽ chạy 1 LẦN DUY NHẤT khi server khởi động.
    Nó tải tất cả 5 file .joblib vào biến 'artifacts'
    """
    print("Khởi động server... đang tải 'dụng cụ'...")
    
    artifacts["model"] = joblib.load("isolation_forest_model.joblib")
    artifacts["scaler"] = joblib.load("scaler.joblib")
    artifacts["ohe"] = joblib.load("one_hot_encoder.joblib")
    artifacts["le"] = joblib.load("label_encoder.joblib")
    artifacts["lists"] = joblib.load("preprocessing_lists.joblib")
    
    print("--- ĐÃ TẢI XONG 5 'DỤNG CỤ' ---")
    print(f"Model: {type(artifacts['model'])}")
    print(f"Scaler: {type(artifacts['scaler'])}")
    print(f"OHE: {type(artifacts['ohe'])}")
    print(f"Lists: {artifacts['lists'].keys()}")
    print("---------------------------------")
    print("Server đã sẵn sàng nhận log!")


class LogRequest(BaseModel):
    logs: List[Dict[str, Any]] 

class PredictionOutput(BaseModel):
    anomaly_score: float  
    is_anomaly: bool      

def preprocess_logs(logs: List[Dict[str, Any]]) -> pd.DataFrame:
    """
    Hàm này nhận log thô từ Logstash và tiền xử lý y hệt
    như các bước đã làm trong Google Colab.
    """

    df = pd.DataFrame(logs)
    
    lists = artifacts["lists"]
    rare_protocols = lists['rare_protocols']
    skewed_cols = lists['skewed_cols']
    numerical_features = lists['numerical_features']
    categorical_features = lists['categorical_features']
    ohe_col_names = lists['ohe_col_names']
    all_feature_names_in_order = lists['all_feature_names_in_order']

    df['service'].replace('-', 'none', inplace=True)
    df['proto'] = df['proto'].apply(lambda x: 'other' if x in rare_protocols else x)


    
    cols_to_transform = [col for col in skewed_cols if col in df.columns]
    df[cols_to_transform] = df[cols_to_transform].apply(np.log1p)
    
    df[numerical_features] = artifacts["scaler"].transform(df[numerical_features])
    
    encoded_data = artifacts["ohe"].transform(df[categorical_features])
    df_encoded = pd.DataFrame(encoded_data, columns=ohe_col_names, index=df.index)
    
    df_processed = pd.concat([df[numerical_features], df_encoded], axis=1)
    
    df_processed = df_processed[all_feature_names_in_order]
    
    return df_processed

@app.post("/predict", response_model=List[PredictionOutput])
async def predict_anomaly(request: LogRequest):
    """
    Đây là "cửa" mà Logstash sẽ gửi log đến.
    Nó nhận log, tiền xử lý, dự đoán và trả về kết quả.
    """
    # 1. Lấy dữ liệu log thô từ request
    logs_raw = request.logs
    
    # 2. Tiền xử lý (Preprocessing)
    df_processed = preprocess_logs(logs_raw)
    
    # 3. Dự đoán (Prediction)
    model = artifacts["model"]
    # .decision_function() trả về điểm số (càng âm càng bất thường)
    scores = model.decision_function(df_processed)
    # .predict() trả về 1 (bình thường) hoặc -1 (bất thường)
    predictions = model.predict(df_processed)
    
    # 4. Định dạng Kết quả
    results = []
    for score, pred in zip(scores, predictions):
        results.append(
            PredictionOutput(
                anomaly_score=score,
                is_anomaly= (pred == -1) # True nếu là -1, False nếu là 1
            )
        )
    
    return results

# --- CHẠY SERVER (NẾU CHẠY TRỰC TIẾP FILE NÀY) ---
if __name__ == "__main__":
    # Đây là lệnh để chạy server khi bạn gõ "python server.py"
    # --host 0.0.0.0 nghĩa là nó chấp nhận kết nối từ bên ngoài (Logstash)
    uvicorn.run(app, host="0.0.0.0", port=8000)