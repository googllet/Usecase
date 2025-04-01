from flask import Flask, render_template, request
import joblib
import pandas as pd
import numpy as np

app = Flask(__name__)
model = joblib.load("random_forest_model_latest.pkl")  # โหลดโมเดล

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # รับค่าจากฟอร์ม
        input_data = {
            'Total_Amount': float(request.form['Total_Amount']),
            'Total_Amount_to_Repay': float(request.form['Total_Amount_to_Repay']),
            'Amount_Funded_By_Lender': float(request.form['Amount_Funded_By_Lender']),
            # เพิ่มฟีเจอร์อื่นๆ ที่ใช้ด้วย ถ้ามี
        }

        # เตรียมข้อมูลทำนาย (ต้อง log1p ด้วยถ้าใช้แบบในโค้ด)
        input_df = pd.DataFrame([input_data])
        input_df[["Total_Amount", "Total_Amount_to_Repay", "Amount_Funded_By_Lender"]] = \
            np.log1p(input_df[["Total_Amount", "Total_Amount_to_Repay", "Amount_Funded_By_Lender"]])

        # ทำนาย
        prediction = model.predict(input_df)[0]

        return render_template('index.html', prediction_text=f'ผลการทำนาย: {prediction}')
    
    except Exception as e:
        return render_template('index.html', prediction_text=f'เกิดข้อผิดพลาด: {e}')

if __name__ == '__main__':
    app.run(debug=True)
