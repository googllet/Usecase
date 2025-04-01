# 🧠 Credit Scoring Web App for Airtime Loans

> สร้างระบบช่วยทำนายความน่าเชื่อถือของผู้ขอสินเชื่อระยะสั้น (Airtime Loans) โดยใช้โมเดล Machine Learning ตามแนวทางจากงานวิจัย

## 📌 Use Case นี้ทำอะไร
ระบบนี้จำลองกระบวนการประเมินความเสี่ยงของผู้ขอสินเชื่อ โดย:
- รับข้อมูลจากผู้ใช้งาน (ผ่านหน้าเว็บ)
- ประมวลผลข้อมูลและแปลงให้อยู่ในรูปที่โมเดลเข้าใจ
- ใช้โมเดล **Random Forest** ที่ฝึกจากข้อมูลจริงในการตัดสินใจ
- แสดงผลการทำนายว่า “มีแนวโน้มชำระเงินหรือไม่”

## 🔍 อ้างอิงจากงานวิจัย
**Paper**: [Use of Machine Learning Techniques to Create a Credit Score Model for Airtime Loans](https://www.mdpi.com/1911-8074/13/8/180)  
**Publisher**: MDPI  
**ปี**: 2021

## 🧾 ข้อมูลที่ใช้ในการฝึกโมเดล
- `Total_Amount` — จำนวนเงินที่ขอ
- `Total_Amount_to_Repay` — จำนวนที่ต้องชำระคืน
- `Amount_Funded_By_Lender` — จำนวนเงินที่ผู้ให้กู้อนุมัติ
- `target` — ตัวแปรเป้าหมาย (0 = ไม่ชำระคืน, 1 = ชำระคืน)

ใช้การจัดการข้อมูล เช่น:
- Log transformation แก้ skew
- ลบ outliers ด้วย IQR
- ทำ class balancing ด้วย `class_weight`

## ⚙️ วิธีใช้งานระบบ

### 1. สร้าง virtual environment (venv)
```bash
python -m venv venv
venv\\Scripts\\activate  # Windows
2. ติดตั้งไลบรารี
pip install -r requirements.txt
3. เริ่มต้นเซิร์ฟเวอร์ Flask
python app.py
4. เข้าใช้งานในเบราว์เซอร์
เปิดเบราว์เซอร์ที่ http://localhost:5000 แล้วกรอกข้อมูลเพื่อทำนาย

📁 โครงสร้างโปรเจกต์
Usecase/
├── app.py                        # Flask backend
├── random_forest_model_latest.pkl  # โมเดลที่เทรนแล้ว
├── requirements.txt             # ไลบรารีที่ต้องใช้
├── templates/
│   └── index.html               # หน้าเว็บ HTML
└── README.md                    # เอกสารอธิบายโปรเจกต์นี้

📌 หมายเหตุ
โมเดลนี้สามารถปรับขยายได้ เช่น เพิ่มฟีเจอร์, เชื่อมฐานข้อมูล, เพิ่มระบบอนุมัติอัตโนมัติ

ระบบนี้ทำงานบนข้อมูลจำลองเพื่อการศึกษา


