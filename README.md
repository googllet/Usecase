# 📘 Credit Scoring Web App (Usecase)

เว็บแอปนี้สร้างขึ้นเพื่อจำลองระบบทำนายความน่าเชื่อถือของผู้ขอสินเชื่อ Airtime โดยใช้ Machine Learning อ้างอิงแนวคิดจากงานวิจัยของ MDPI:

🔗 [Use of Machine Learning Techniques to Create a Credit Score Model for Airtime Loans](https://www.mdpi.com/1911-8074/13/8/180)

---

## 🧠 รายละเอียดของ Use Case

ระบบสามารถ:
- รับข้อมูลจากผู้ใช้ผ่านฟอร์ม HTML
- ประมวลผลข้อมูลที่ได้รับ (Log-transform)
- ส่งข้อมูลเข้าโมเดล Random Forest ที่เทรนแล้ว
- แสดงผลการทำนายว่า ผู้ขอสินเชื่อนั้นน่าเชื่อถือหรือไม่ (มีแนวโน้มจ่ายคืนหรือไม่)

---

## 📁 โครงสร้างโปรเจกต์

```
Usecase/
├── app.py                          # Flask backend
├── requirements.txt               # ไลบรารีที่ต้องติดตั้ง
├── random_forest_model_latest.pkl # โมเดล ML ที่เทรนไว้แล้ว
├── templates/
│   └── index.html                 # หน้าเว็บ HTML
└── README.md                      # คำอธิบายโปรเจกต์นี้
```

---

## ⚙️ ขั้นตอนการใช้งาน

### 1. ✅ ติดตั้ง Python เวอร์ชัน 3.11
> แนะนำให้ใช้ Python 3.11.x เพื่อความเข้ากันได้กับโมเดล `.pkl`

สามารถดาวน์โหลดได้จาก: https://www.python.org/downloads/release/python-3119/

### 2. ✅ สร้าง Virtual Environment (venv)
```bash
python -m venv venv
```

### 3. ✅ เปิดใช้งาน venv

- บน **Windows**:
```bash
venv\Scripts\activate
```

- บน **macOS / Linux**:
```bash
source venv/bin/activate
```

### 4. ✅ ติดตั้งไลบรารีที่ใช้
```bash
pip install -r requirements.txt
```

### 5. ✅ รันเว็บแอป
```bash
python app.py
```

เมื่อรันสำเร็จ ให้เปิดเบราว์เซอร์ที่:

```
http://localhost:5000
```

---

## 🧪 รายละเอียดของโมเดลที่ใช้

- โมเดล: RandomForestClassifier
- ข้อมูล: `balanced_train.csv` (ทำ Log-transform + remove outliers แล้ว)
- เทรนใหม่บน Python 3.11 + scikit-learn 1.6.1
- ใช้ `class_weight="balanced"` เพื่อแก้ class imbalance
- บันทึกไว้ในไฟล์ `random_forest_model_latest.pkl`

---

## 📬 ติดต่อหรือขอข้อมูลเพิ่ม

ผู้พัฒนา: [@googllet](https://github.com/googllet)

หากต้องการให้เพิ่มฟีเจอร์ใหม่ หรือปรับ UI เพิ่มเติม สามารถ fork โปรเจกต์นี้ได้เลย 😄