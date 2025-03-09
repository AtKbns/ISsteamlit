import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
import streamlit as st
import gdown  # ต้อง import gdown ด้วย

# โหลด Dataset
data = pd.read_csv('loan_approvals.csv')

# แทนที่ค่าที่หายไป (nan)
data['Gender'] = data['Gender'].fillna('Unknown')
data['Married'] = data['Married'].fillna('Unknown')
data['Dependents'] = data['Dependents'].replace({'3+': 3})  # แปลง 3+ เป็น 3
data['Dependents'] = data['Dependents'].fillna('0')
data['Education'] = data['Education'].fillna('Unknown')
data['Loan_Status'] = data['Loan_Status'].map({'Yes': 1, 'No': 0})

# ใช้ LabelEncoder สำหรับข้อมูลหมวดหมู่
label_encoders = {}
categorical_cols = ['Gender', 'Married', 'Dependents', 'Education']
for col in categorical_cols:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col].astype(str))  # แปลงเป็น str ก่อนเข้ารหัส
    label_encoders[col] = le

# แยก Features (X) และ Target (y)
X = data.drop(columns=['Loan_Status'])
y = data['Loan_Status']

# แบ่งข้อมูลเป็น Training และ Test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ปรับขนาดข้อมูล (Standardize)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# สร้างโมเดล SVM
svm = SVC(class_weight='balanced')

# ฝึกโมเดล SVM
svm.fit(X_train, y_train)

# สร้างหน้า UI ด้วย Streamlit
st.title("Loan Approval Prediction")

# สร้างแท็บ (Tab)
tabs = ['Machine learning', 'Neural network', 'Demo Machine learning', 'Demo Neural network']
selected_tab = st.selectbox('Choose a Tab', tabs)

# แสดงผลตามแท็บที่เลือก
if selected_tab == 'Machine learning':
    st.header("Machine Learning")
    
    st.subheader("ข้อมูลการอนุมัติสินเชื่อ (Loan Approval Data)")
    st.write("""
        เราจะใช้ข้อมูลการอนุมัติสินเชื่อ (Loan Approval Data) ที่เก็บรายละเอียดเกี่ยวกับผู้สมัครสินเชื่อจากธนาคาร
        โดยข้อมูลในชุดนี้จะถูกใช้ในการสร้างโมเดล Machine Learning เพื่อทำนายว่า ผู้สมัครสินเชื่อจะได้รับการอนุมัติหรือไม่
        โดยใช้ฟีเจอร์ต่าง ๆ ดังนี้:
    """)
    
    st.write("""
        - **Gender**: เพศของผู้สมัคร
        - **Married**: สถานภาพการสมรส
        - **Dependents**: จำนวนคนที่พึ่งพิงทางการเงินจากผู้สมัคร
        - **Education**: ระดับการศึกษา
        - **Income**: รายได้ของผู้สมัคร
        - **Loan_Amount**: จำนวนเงินที่ขอสินเชื่อ
        - **Loan_Status**: ผลลัพธ์การอนุมัติสินเชื่อ (เป้าหมาย)
    """)

    st.write("""
    ก่อนอื่นเราจะเริ่มจากการโหลดข้อมูลจากไฟล์ `loan_approvals.csv` ซึ่งในไฟล์นี้จะมีคอลัมน์ที่เป็นข้อมูลหมวดหมู่ เช่น Gender, Married และคอลัมน์ที่เป็นข้อมูลเชิงตัวเลข เช่น Income, Loan_Amount เพื่อที่จะทำความเข้าใจข้อมูลที่เรามี ก่อนที่จะทำการประมวลผลข้อมูลเพิ่มเติม
    """)

    # URL ของไฟล์ที่แชร์จาก Google Drive
    url = 'https://drive.google.com/uc?export=download&id=1QGEXA89PMyjqtR7rbFU41ev4M5vVMq-B'

    # ดาวน์โหลดไฟล์จาก Google Drive
    gdown.download(url, 'loan_approvals.csv', quiet=False)

    # ปุ่มดาวน์โหลดใน Streamlit
    with open('loan_approvals.csv', 'rb') as file:
        st.download_button(
            label="ดาวน์โหลดไฟล์ loan_approvals.csv",
            data=file,
            file_name="loan_approvals.csv",
            mime="text/csv"
        )

    st.code("""
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
import ipywidgets as widgets
from IPython.display import display

# โหลด Dataset
data = pd.read_csv('loan_approvals.csv')

# แสดงข้อมูลบางส่วนเพื่อดูว่ามีอะไรบ้าง
data.head()
    """, language='python')

    st.write("ข้อมูลบางส่วนจาก dataset:")
    st.dataframe(data.head())
    st.write("""
    ต่อไปเราจะจัดการค่าที่หายไป (Missing Values) ในข้อมูล โดยเติมค่าในคอลัมน์หมวดหมู่ เช่น **Gender**, **Married**, **Education** ด้วย 'Unknown' และเติมค่าในคอลัมน์ตัวเลข เช่น **Loan_Amount** ด้วยค่าเฉลี่ยของคอลัมน์นั้น เพื่อให้ข้อมูลสมบูรณ์และพร้อมใช้ในการสร้างโมเดล
""")
    st.code("""
# แทนที่ค่าที่หายไป (Missing Values)
data.fillna({'Gender': 'Unknown', 'Married': 'Unknown', 'Dependents': '0'}, inplace=True)
data.fillna(data.mean(numeric_only=True), inplace=True)

# ตรวจสอบว่าค่าที่หายไปได้รับการจัดการเรียบร้อยแล้ว
data.isnull().sum()
""", language='python')

# ตรวจสอบค่าที่หายไป
    st.write("ผลลัพธ์การตรวจสอบค่าที่หายไปหลังการจัดการ:")
    st.write(data.isnull().sum())


elif selected_tab == 'Neural network':
    st.header("Neural Network Overview")
    st.write("""
        This section will explain how a **Neural Network** can be used to predict loan approval.
        We will be discussing how Neural Networks differ from traditional machine learning models
        and the advantages they provide in complex decision-making tasks.
    """)

elif selected_tab == 'Demo Machine learning':
    st.header("Machine Learning Demo")
    st.subheader("Support Vector Machine")
    # รับข้อมูลจากผู้ใช้
    gender = st.selectbox('Gender', ['Male', 'Female'])
    marital_status = st.selectbox('Marital Status', ['Yes', 'No'])
    education = st.selectbox('Education Level', ['Graduate', 'Not Graduate'])
    dependents = st.selectbox('Dependents', ['0', '1', '2', '3+'])
    income = st.number_input("Income", min_value=0)
    loan_amount = st.number_input("Loan Amount", min_value=0)

    # ฟังก์ชันสำหรับการเตรียมข้อมูลจากผู้ใช้
    def get_user_input(gender, marital_status, education, dependents, income, loan_amount):
        try:
            gender = label_encoders['Gender'].transform([gender])[0]
        except ValueError:
            gender = label_encoders['Gender'].transform(['Unknown'])[0]
        
        try:
            marital_status = label_encoders['Married'].transform([marital_status])[0]
        except ValueError:
            marital_status = label_encoders['Married'].transform(['Unknown'])[0]
        
        dependents = '3' if dependents == '3+' else dependents
        try:
            dependents = label_encoders['Dependents'].transform([str(dependents)])[0]
        except ValueError:
            dependents = label_encoders['Dependents'].transform(['0'])[0]
        
        try:
            education = label_encoders['Education'].transform([education])[0]
        except ValueError:
            education = label_encoders['Education'].transform(['Unknown'])[0]
        
        # เตรียมข้อมูลสำหรับการทำนาย
        user_data = np.array([[gender, marital_status, dependents, education, income, loan_amount]])
        user_data = scaler.transform(user_data)  # ปรับขนาดข้อมูล
        prediction = svm.predict(user_data)
        
        return prediction

    # ทำนายผลเมื่อผู้ใช้คลิกปุ่ม
    if st.button('Predict'):
        prediction = get_user_input(gender, marital_status, education, dependents, income, loan_amount)
        if prediction == 1:
            st.success("Loan Approved")
        else:
            st.error("Loan Denied")


import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from keras.optimizers import SGD
from PIL import Image
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt  # สำหรับการแสดงกราฟ

# โหลดโมเดลและข้อมูลจาก session
def load_model_and_data():
    if 'model' not in st.session_state:
        # โหลดชุดข้อมูล MNIST สำหรับการฝึกโมเดล Neural Network
        (x_train, y_train), (x_test, y_test) = mnist.load_data()

        # ทำ Normalization
        x_train, x_test = x_train / 255.0, x_test / 255.0

        # แปลงข้อมูลเป็น (28, 28, 1) สำหรับ Conv2D
        x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
        x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)

        # One-hot encode labels
        y_train = to_categorical(y_train, 10)
        y_test = to_categorical(y_test, 10)

        # สร้างโมเดล
        model = Sequential([
            Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
            MaxPooling2D(2, 2),
            Conv2D(64, (3, 3), activation='relu'),
            MaxPooling2D(2, 2),
            Flatten(),
            Dense(128, activation='relu'),
            Dense(10, activation='softmax')
        ])

        # ตั้งค่า optimizer และ compile โมเดล
        model.compile(optimizer=SGD(learning_rate=0.01, momentum=0.9), 
                      loss='categorical_crossentropy', 
                      metrics=['accuracy'])

        # ฝึกโมเดล และบันทึกข้อมูล history
        history = model.fit(x_train, y_train, epochs=2)

        # ทดสอบโมเดล
        test_loss, test_acc = model.evaluate(x_test, y_test)
        print(f"Test accuracy: {test_acc}")

        # เก็บโมเดลใน session
        st.session_state.model = model
        st.session_state.history = history
        st.session_state.x_test = x_test
        st.session_state.y_test = y_test
    else:
        model = st.session_state.model
        history = st.session_state.history
        x_test = st.session_state.x_test
        y_test = st.session_state.y_test

    return model, history, x_test, y_test

# เมื่อโมเดลฝึกเสร็จแล้ว ใช้ในการทำนาย
if selected_tab == 'Demo Neural network':
    model, history, x_test, y_test = load_model_and_data()

    st.header("Neural Network Demo")

    st.write(""" 
        ในส่วนนี้เราจะแสดงตัวอย่างการทำนายจากโมเดล Neural Network ที่ได้ฝึกกับข้อมูล MNIST
        โมเดลนี้สามารถทำนายตัวเลขจากภาพที่อัปโหลดเข้ามาได้
    """)

    # แสดงกราฟ Accuracy และ Loss ก่อนการอัปโหลด
    st.header("Training History")

    # จัดการแสดงกราฟแบบคอลัมน์
    col1, col2 = st.columns(2)

    with col1:
        # กราฟ Accuracy
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(history.history['accuracy'], label='Training Accuracy')
        ax.set_title('Model Accuracy')
        ax.set_xlabel('Epochs')
        ax.set_ylabel('Accuracy')
        ax.legend()
        st.pyplot(fig)

    with col2:
        # กราฟ Loss
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(history.history['loss'], label='Training Loss')
        ax.set_title('Model Loss')
        ax.set_xlabel('Epochs')
        ax.set_ylabel('Loss')
        ax.legend()
        st.pyplot(fig)

    # แสดงข้อความกับการจัดวางในคอลัมน์
    col1, col2 = st.columns([2, 3])

    with col1:
        st.image("https://upload.wikimedia.org/wikipedia/commons/a/a5/MnistExample.png", caption="ตัวอย่างข้อมูลจาก MNIST", use_container_width=True)

    with col2:
        st.write("""
            โมเดลนี้ถูกฝึกด้วยข้อมูลจาก MNIST ซึ่งเป็นชุดข้อมูลของตัวเลขที่เขียนด้วยมือ
            โดยโมเดลนี้สามารถทำนายตัวเลขจากภาพที่คุณอัปโหลดได้
        """)

    # อัปโหลดรูปภาพจากผู้ใช้
    uploaded = st.file_uploader("อัปโหลดรูปภาพที่ต้องการทำนาย", type=["jpg", "png", "jpeg"])

    if uploaded is not None:
        # แปลงภาพที่อัปโหลด
        img = Image.open(uploaded).convert('L')  # แปลงเป็น grayscale
        img = img.resize((28, 28))  # ปรับขนาดเป็น 28x28
        img_array = np.array(img)  # แปลงเป็น numpy array
        img_array = img_array / 255.0  # Normalization
        img_array = img_array.reshape(1, 28, 28, 1)  # ปรับขนาดข้อมูลเป็น (1, 28, 28, 1)

        # ทำนายผลจากโมเดล
        prediction = model.predict(img_array)
        predicted_label = np.argmax(prediction)

        # แสดงผลลัพธ์การทำนาย  
        st.image(img, caption='Uploaded Image', use_container_width=True)  # แก้เป็น use_container_width=True
        st.markdown(f"<h1 style='text-align: center; font-size: 60px;'>Predicted Number: {predicted_label}</h1>", unsafe_allow_html=True)





