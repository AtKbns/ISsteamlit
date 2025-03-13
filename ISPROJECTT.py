# Load Dataset
def load_data():
    data = pd.read_csv('loan_approvals.csv')
    # Handle missing values
    data['Gender'] = data['Gender'].fillna('Unknown')
    data['Married'] = data['Married'].fillna('Unknown')
    data['Dependents'] = data['Dependents'].replace({'3+': 3})  # Convert '3+' to 3
    data['Dependents'] = data['Dependents'].fillna('0')
    data['Education'] = data['Education'].fillna('Unknown')
    data['Loan_Status'] = data['Loan_Status'].map({'Yes': 1, 'No': 0})

    # Apply Label Encoding for categorical features
    label_encoders = {}
    categorical_cols = ['Gender', 'Married', 'Dependents', 'Education']
    for col in categorical_cols:
        le = LabelEncoder()
        data[col] = le.fit_transform(data[col].astype(str))
        label_encoders[col] = le

    return data

data = load_data()

# Display sample data
st.write(data.head())

# Prepare Features and Target
X = data.drop('Loan_Status', axis=1)  # Features
y = data['Loan_Status']  # Target
X = X.fillna(X.mean())  # Handle missing numerical values

# Normalize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Model selection in Streamlit UI
model_choice = st.radio("Choose the model:", ('SVM', 'KNN', 'Random Forest'))

# Create and train the selected model
if model_choice == 'SVM':
    model = SVC()
elif model_choice == 'KNN':
    model = KNeighborsClassifier()
elif model_choice == 'Random Forest':
    model = RandomForestClassifier()

# Train the model
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Show accuracy result
accuracy = accuracy_score(y_test, y_pred)
st.write(f"Accuracy: {accuracy*100:.2f}%")

# UI Tabs for navigation
tabs = ['Machine learning', 'Neural network', 'Demo Machine learning', 'Demo Neural network']
selected_tab = st.selectbox('Choose a Tab', tabs)



# แสดงผลตามแท็บที่เลือก
if selected_tab == 'Machine learning':
    if "svm_model" not in st.session_state:
        st.session_state.svm_model = SVC(kernel='rbf', C=1.0, gamma='scale')  # สร้างโมเดล SVM ด้วยค่าเริ่มต้น
        st.session_state.svm_model.fit(X_train, y_train)  # ฝึกโมเดล SVM ด้วยข้อมูลฝึก

    model = st.session_state.svm_model  # โหลดโมเดลที่ฝึกแล้วจาก session_state
    st.header("Machine Learning")

    # แนะนำข้อมูลการอนุมัติสินเชื่อ
    st.subheader("ข้อมูลการอนุมัติสินเชื่อ (Loan Approval Data)")
    st.write("""
    ข้อมูลที่เราจะใช้ในโครงการนี้เป็นข้อมูลการอนุมัติสินเชื่อ ซึ่งบ่งบอกว่าผู้สมัครสินเชื่อได้รับการอนุมัติหรือไม่
    โดยเราจะใช้ฟีเจอร์ต่างๆ ในการทำนายผลลัพธ์ดังกล่าว โดยมีข้อมูลในรูปแบบต่างๆ ดังนี้:
    - **Gender**: เพศของผู้สมัคร (ชาย/หญิง)
    - **Married**: สถานภาพการสมรส (แต่งงาน/ไม่ได้แต่งงาน)
    - **Dependents**: จำนวนบุคคลที่ผู้สมัครพึ่งพา (เช่น เด็ก หรือผู้ที่ต้องดูแล)
    - **Education**: ระดับการศึกษา (ปริญญาตรี/สูงกว่า)
    - **Income**: รายได้ของผู้สมัคร
    - **Loan_Amount**: จำนวนเงินที่ผู้สมัครขอสินเชื่อ
    - **Loan_Status**: การอนุมัติสินเชื่อ (ผ่าน/ไม่ผ่าน)
    """)

    # การดาวน์โหลดไฟล์ข้อมูล
    st.write("คุณสามารถดาวน์โหลดข้อมูลการอนุมัติสินเชื่อได้ที่นี่:")

# สร้างลิงก์สำหรับคลิกไปยัง Google Drive
    drive_link = 'https://drive.google.com/drive/u/0/folders/1M49d6uX87fUK6YtjgN-TmAFZ-hjZ33wb'

# ใช้ st.markdown เพื่อแสดงลิงก์
    st.markdown(f"[คลิกที่นี่เพื่อเปิด Google Drive]( {drive_link} )")

    # แสดงตัวอย่างโค้ดการโหลดข้อมูล
    st.code("""
import pandas as pd
data = pd.read_csv('loan_approvals.csv')
data.head()  # แสดงข้อมูลบางส่วน
    """, language='python')

    # การจัดการ Missing Values
    st.write("เราต้องจัดการกับค่าที่หายไป (Missing Values) เพื่อให้ข้อมูลมีความสมบูรณ์")
    st.code("""
# แทนที่ค่าที่หายไปด้วยค่าเฉลี่ยสำหรับข้อมูลตัวเลข
data.fillna(data.mean(numeric_only=True), inplace=True)

# แทนที่ค่าที่หายไปในคอลัมน์ที่เป็นหมวดหมู่ (เช่น Gender, Married, Dependents) ด้วยคำว่า 'Unknown'
data.fillna({'Gender': 'Unknown', 'Married': 'Unknown', 'Dependents': '0'}, inplace=True)
    """, language='python')

    # การแปลงข้อมูลหมวดหมู่ (Categorical Data) เป็นตัวเลข
    st.write("จากนั้นเราจะใช้ Label Encoding เพื่อแปลงข้อมูลหมวดหมู่เป็นตัวเลข ซึ่งจะทำให้โมเดลสามารถประมวลผลได้")
    st.code("""
from sklearn.preprocessing import LabelEncoder
categorical_cols = ['Gender', 'Married', 'Dependents', 'Education']

# แปลงคอลัมน์ที่เป็นหมวดหมู่ให้เป็นตัวเลข
for col in categorical_cols:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col])
    """, language='python')

    # แยกข้อมูลเป็น Features และ Target
    st.write("หลังจากแปลงข้อมูลเสร็จ เราจะแยกข้อมูลออกเป็น Features (ข้อมูลที่ใช้ทำนาย) และ Target (ผลลัพธ์ที่เราต้องการทำนาย)")
    st.code("""
# Features คือข้อมูลทั้งหมดยกเว้นคอลัมน์ที่เป็นผลลัพธ์ (Loan_Status)
X = data.drop(columns=['Loan_Status'])

# Target คือคอลัมน์ที่เป็นผลลัพธ์ (Loan_Status) ซึ่งเป็นข้อมูลที่เราต้องการทำนาย
y = LabelEncoder().fit_transform(data['Loan_Status'])  # แปลง Loan_Status เป็นตัวเลข
    """, language='python')

    # การแบ่งข้อมูลสำหรับการฝึกและทดสอบ
    st.write("หลังจากที่เราแยก Features และ Target แล้ว เราจะต้องแบ่งข้อมูลออกเป็นข้อมูลสำหรับการฝึก (Training Data) และข้อมูลสำหรับทดสอบ (Test Data)")
    st.code("""
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    """, language='python')

    # การปรับขนาดข้อมูล (Feature Scaling)
    st.write("ในการฝึกโมเดล เราจะปรับขนาดข้อมูลให้มีค่าระหว่าง -1 และ 1 เพื่อให้การฝึกเป็นไปได้อย่างมีประสิทธิภาพมากขึ้น")
    st.code("""
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)  # ปรับขนาดข้อมูลสำหรับชุดฝึก
X_test = scaler.transform(X_test)  # ปรับขนาดข้อมูลสำหรับชุดทดสอบ
    """, language='python')

    # สร้างโมเดล SVM
    st.write("ตอนนี้เราจะสร้างโมเดล SVM (Support Vector Machine) เพื่อทำการทำนายผลลัพธ์ว่า ผู้สมัครจะได้รับการอนุมัติสินเชื่อหรือไม่")
    st.code("""
from sklearn.svm import SVC
svm = SVC(class_weight='balanced')  # ตั้งค่าพารามิเตอร์สำหรับ SVM
svm.fit(X_train, y_train)  # ฝึกโมเดลด้วยข้อมูลชุดฝึก
y_pred = svm.predict(X_test)  # ทำนายผลลัพธ์ด้วยข้อมูลชุดทดสอบ
print("Accuracy:", accuracy_score(y_test, y_pred))  # แสดงค่าความแม่นยำ
    """, language='python')

    # ใช้ GridSearchCV สำหรับการปรับพารามิเตอร์ของ SVM
    st.write("เราสามารถใช้ GridSearchCV เพื่อลองค่าพารามิเตอร์ต่างๆ เพื่อหาค่าที่ดีที่สุดสำหรับ SVM และ KNN")
    st.code("""
from sklearn.model_selection import GridSearchCV
param_grid_svm = {'C': [0.1, 1, 10], 'kernel': ['linear', 'rbf'], 'gamma': ['scale', 'auto']}
grid_search_svm = GridSearchCV(SVC(), param_grid_svm, cv=5)  # ใช้ Cross Validation เพื่อหาค่าพารามิเตอร์ที่ดีที่สุด
grid_search_svm.fit(X_train, y_train)  # ฝึกโมเดล
print("Best Parameters for SVM:", grid_search_svm.best_params_)  # แสดงค่าพารามิเตอร์ที่ดีที่สุด
    """, language='python')

    # เปรียบเทียบผลลัพธ์จากโมเดลต่างๆ
    st.write("สุดท้าย เราสามารถเปรียบเทียบผลลัพธ์ของโมเดลต่างๆ เพื่อดูว่าโมเดลใดทำงานได้ดีที่สุด")
    st.code("""
models = {'SVM': grid_search_svm.best_estimator_}
for name, model in models.items():
    model.fit(X_train, y_train)  # ฝึกโมเดล
    y_pred = model.predict(X_test)  # ทำนายผลลัพธ์
    print(f"Model: {name}")
    print("Accuracy:", accuracy_score(y_test, y_pred))  # แสดงความแม่นยำ
    print("Classification Report:\n", classification_report(y_test, y_pred))  # แสดงรายงานความแม่นยำ
    """, language='python')



elif selected_tab == 'Neural network':
    st.header("🧠 Neural Network Model - MNIST")

    st.subheader("📌 1️⃣ โหลดข้อมูล MNIST")
    st.write("ทำการโหลดชุดข้อมูล **MNIST** ซึ่งเป็นรูปภาพตัวเลข 0-9 ขนาด 28x28 พิกเซล")
    st.code("""
    from keras.datasets import mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    """, language="python")

    st.subheader("📌 2️⃣ ทำ Normalization ข้อมูล")
    st.write("แปลงค่าพิกเซลจาก **0-255** ให้เป็น **0-1** เพื่อช่วยให้โมเดลเรียนรู้ได้ดีขึ้น")
    st.code("""
    x_train, x_test = x_train / 255.0, x_test / 255.0
    """, language="python")

    st.subheader("📌 3️⃣ ปรับโครงสร้างข้อมูล")
    st.write("แปลงข้อมูลเป็น `(28, 28, 1)` เพื่อให้เหมาะกับ Convolutional Layers และทำ One-hot Encoding ของ Labels")
    st.code("""
    x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
    x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)

    from keras.utils import to_categorical
    y_train = to_categorical(y_train, 10)
    y_test = to_categorical(y_test, 10)
    """, language="python")

    st.subheader("📌 4️⃣ สร้างโครงสร้างโมเดล CNN")
    st.write("สร้างโมเดล **Convolutional Neural Network (CNN)** โดยมี 2 ชั้นของ Convolutional Layers และ MaxPooling")
    st.code("""
    from keras.models import Sequential
    from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        MaxPooling2D(2, 2),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        Flatten(),
        Dense(128, activation='relu'),
        Dense(10, activation='softmax')
    ])
    """, language="python")

    st.subheader("📌 5️⃣ ตั้งค่า Optimizer และ Compile โมเดล")
    st.write("ใช้ `SGD` (Stochastic Gradient Descent) พร้อม `Momentum` และกำหนด Loss Function เป็น `categorical_crossentropy`")
    st.code("""
    from keras.optimizers import SGD

    model.compile(optimizer=SGD(learning_rate=0.01, momentum=0.9),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    """, language="python")

    st.subheader("📌 6️⃣ ฝึกโมเดล")
    st.write("ใช้ข้อมูล `x_train` และ `y_train` ทำการ Train โมเดลเป็นเวลา 2 Epochs")
    st.code("""
    model.fit(x_train, y_train, epochs=2)
    """, language="python")

    st.subheader("📌 7️⃣ ทดสอบการพยากรณ์ตัวเลขจากภาพที่อัปโหลด")
    st.write("อัปโหลดภาพ, แปลงให้เป็น 28x28, ทำ Normalization และใช้โมเดลทำนายตัวเลข")
    st.code("""
    from google.colab import files
    from PIL import Image
    import numpy as np
    import matplotlib.pyplot as plt

    uploaded = files.upload()

    for fn in uploaded.keys():
        img = Image.open(fn).convert('L')  # แปลงเป็น grayscale
        img = img.resize((28, 28))
        img_array = np.array(img) / 255.0
        img_array = img_array.reshape(1, 28, 28, 1)

        prediction = model.predict(img_array)
        predicted_label = np.argmax(prediction)

        print(f"Prediction: {predicted_label}")
        plt.imshow(img, cmap='gray')
        plt.title(f"Predicted Number: {predicted_label}")
        plt.show()
    """, language="python")

    st.write("✅ **โมเดลนี้สามารถทำนายตัวเลขจากภาพที่อัปโหลดได้! 🎯**")

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





