📘 Credit Scoring Web App (Based on ML Model from MDPI Paper)
🔍 Use Case:
ระบบทำนายความน่าเชื่อถือของผู้ขอสินเชื่อ Airtime Loans โดยใช้เทคนิค Machine Learning
อ้างอิงจากเปเปอร์:
“Use of Machine Learning Techniques to Create a Credit Score Model for Airtime Loans”
🔗 MDPI Paper Link

🧠 รายละเอียดของสิ่งที่ระบบนี้ทำ
รับข้อมูลการขอสินเชื่อ เช่น จำนวนที่ขอ, จำนวนต้องชำระ, เงินที่ได้รับจากผู้ให้กู้

ใช้โมเดล Random Forest ที่ถูกเทรนจากข้อมูลจริงในการประเมิน

แสดงผลการทำนายว่า “ผู้ขอนั้นมีแนวโน้มจะจ่ายคืนหรือไม่”

ทำเป็น Web App แบบง่ายๆ ด้วย HTML + Flask

⚙️ ขั้นตอนการพัฒนา
1. เตรียมข้อมูล (balanced_train.csv)
ตรวจสอบ Skewness และแปลงด้วย Log1p

ลบ Outliers ด้วย IQR Method

ใช้ Feature Engineering ตามเปเปอร์

จัด balance class ด้วย class_weight

2. สร้างและฝึกโมเดล
ใช้ 3 โมเดล: Logistic Regression, Decision Tree, Random Forest

ใช้ Random Forest เป็นโมเดลที่ดีที่สุด

บันทึกโมเดลเป็น .pkl เพื่อใช้งานใน Web App

3. สร้าง Web App ด้วย Flask + HTML
หน้าเว็บมีฟอร์มให้กรอกข้อมูล

ส่งข้อมูลไปยัง backend Flask

ทำนายผลแล้วแสดงออกมา

📁 โครงสร้างไฟล์โปรเจกต์
bash
คัดลอก
แก้ไข
credit-scoring-app/
├── app.py                      # Flask Backend
├── random_forest_model.pkl     # โมเดลที่เทรนแล้ว
├── requirements.txt            # รายการไลบรารีที่ต้องติดตั้ง
└── templates/
    └── index.html              # หน้าเว็บ HTML
🖥️ วิธีใช้งานโปรเจกต์นี้
1. สร้างและเปิดใช้งาน venv
bash
คัดลอก
แก้ไข
python3.11 -m venv venv
venv\Scripts\activate           # (Windows)
2. ติดตั้ง dependencies
bash
คัดลอก
แก้ไข
pip install -r requirements.txt
3. รันเว็บแอป
bash
คัดลอก
แก้ไข
python app.py
เปิดเบราว์เซอร์ที่ http://localhost:5000

📌 หมายเหตุ
ใช้ข้อมูลจำลองในการฝึกโมเดล

ใช้โมเดลต้นแบบตามแนวคิดจากงานวิจัยจริง

สามารถขยายต่อได้ เช่น เพิ่มฟีเจอร์, ตกแต่ง UI, เพิ่มการอัปโหลดไฟล์ CSV
