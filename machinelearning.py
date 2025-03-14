# -*- coding: utf-8 -*-
"""machinelearning

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1y49vIPVROmAcPDltVoSJThRYDkOT7Uyf

### ผมได้เริ่มหาข้อมูลมาจาก CHAT GPT
## เนื้อหาคือเราจะใช้ ข้อมูลการอนุมัติสินเชื่อ (loan approval data) ที่เก็บรายละเอียดเกี่ยวกับผู้สมัครสินเชื่อจากธนาคาร โดยข้อมูลในชุดนี้จะถูกใช้ในการสร้างโมเดล Machine Learning เพื่อทำนายว่า ผู้สมัครสินเชื่อจะได้รับการอนุมัติหรือไม่โดยใช้ฟีเจอร์ต่าง ๆ เช่น:

Gender: เพศของผู้สมัคร

Married: สถานภาพการสมรส

Dependents: จำนวนคนที่พึ่งพิงทางการเงินจากผู้สมัคร

Education: ระดับการศึกษา

Income: รายได้ของผู้สมัคร

Loan_Amount: จำนวนเงินที่ขอสินเชื่อ

Loan_Status: ผลลัพธ์การอนุมัติสินเชื่อ (เป้าหมาย)

# **1. การโหลดข้อมูลและตรวจสอบข้อมูลเบื้องต้น:ก่อนอื่นเราจะเริ่มจากการโหลดข้อมูล loan_approvals.csv ซึ่งประกอบด้วยคอลัมน์ที่เป็นทั้งข้อมูลหมวดหมู่ (เช่น Gender, Married) และข้อมูลเชิงตัวเลข (เช่น Income, Loan_Amount) เพื่อทำความเข้าใจข้อมูลที่เรามี**
"""

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

"""## **2. การจัดการค่าที่หายไป (Missing Values) ข้อมูลบางคอลัมน์อาจมีค่าที่หายไป (Missing Values) ซึ่งจะต้องทำการเติมค่าให้ครบถ้วนก่อน โดยสำหรับคอลัมน์ที่เป็นข้อมูลหมวดหมู่ เช่น Gender, Married, ผมจะเติมค่าที่หายไปด้วย 'Unknown' ส่วนคอลัมน์ตัวเลขเช่น Loan_Amount ผมจะเติมค่าที่หายไปด้วยค่าเฉลี่ย**"""

# แทนที่ค่าที่หายไป (Missing Values)
data.fillna({'Gender': 'Unknown', 'Married': 'Unknown', 'Dependents': '0'}, inplace=True)
data.fillna(data.mean(numeric_only=True), inplace=True)

# ตรวจสอบว่าค่าที่หายไปได้รับการจัดการเรียบร้อยแล้ว
data.isnull().sum()

"""## **3. การแปลงข้อมูลหมวดหมู่เป็นตัวเลข (Encoding Categorical Data):เนื่องจากโมเดล Machine Learning ไม่สามารถใช้ข้อมูลหมวดหมู่ได้โดยตรง เราจึงต้องแปลงข้อมูลเหล่านั้นให้เป็นตัวเลข โดยใช้ LabelEncoder สำหรับข้อมูลหมวดหมู่ เช่น Gender, Married, และ Dependents**"""

from sklearn.preprocessing import LabelEncoder

# สร้าง dictionary สำหรับเก็บ LabelEncoder
label_encoders = {}
categorical_cols = ['Gender', 'Married', 'Dependents', 'Education']

# ใช้ LabelEncoder สำหรับข้อมูลหมวดหมู่
for col in categorical_cols:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col])
    label_encoders[col] = le  # เก็บ encoder ไว้ใช้ตอนหลัง

# แสดงข้อมูลหลังจากการแปลง
data.head()

"""## **4. แยกข้อมูล Features และ Target:ในการสร้างโมเดล เราจะต้องแยกข้อมูลออกเป็น Features (X) ที่เป็นข้อมูลที่ใช้ทำนาย และ Target (y) ซึ่งในกรณีนี้คือ Loan_Status ที่จะเป็นตัวแปรที่เราทำนาย**"""

# แยก Features (X) และ Target (y)
X = data.drop(columns=['Loan_Status'])
y = LabelEncoder().fit_transform(data['Loan_Status'])  # แปลง Loan_Status ให้เป็นตัวเลข

"""## **5. การแบ่งข้อมูลเป็นชุดฝึก (Training) และชุดทดสอบ (Testing):การแบ่งข้อมูลเป็นชุดฝึกและชุดทดสอบเป็นขั้นตอนสำคัญในการป้องกันการ overfitting โดยจะใช้ 80% ของข้อมูลสำหรับการฝึกโมเดลและ 20% สำหรับการทดสอบโมเดล**"""

from sklearn.model_selection import train_test_split

# แบ่งข้อมูลเป็น Training และ Test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# แสดงขนาดของชุดข้อมูล
print("Training data size:", X_train.shape)
print("Test data size:", X_test.shape)

"""## **6. การปรับขนาดข้อมูล (Standardization):เพื่อให้โมเดลทำงานได้ดีขึ้น ผมจะทำการปรับขนาดข้อมูลทั้งหมดให้มีมาตรฐานเดียวกัน (Standardization) โดยใช้ StandardScaler ซึ่งจะช่วยให้โมเดลไม่ถูกกระทบจากขนาดที่แตกต่างของแต่ละฟีเจอร์**"""

from sklearn.preprocessing import StandardScaler

# ปรับขนาดข้อมูล (Standardize)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# แสดงข้อมูลที่ปรับขนาดแล้ว
X_train[:5]

"""## **7. การฝึกโมเดล SVM (Support Vector Machine):ต่อไป ผมจะสร้างโมเดล SVM ซึ่งเป็นหนึ่งในโมเดลที่ได้รับความนิยมในปัญหาการจำแนกประเภท โดยใช้พารามิเตอร์ class_weight='balanced' เพื่อช่วยปรับสมดุลของข้อมูลที่ไม่สมดุล (imbalanced classes)**"""

from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

# สร้างโมเดล SVM ที่มีการตั้งค่า class_weight='balanced'
svm = SVC(class_weight='balanced')

# ฝึกโมเดลด้วยข้อมูลที่เตรียมไว้
svm.fit(X_train, y_train)

# ทำนายผล
y_pred = svm.predict(X_test)

# แสดงผลลัพธ์
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

"""## **8. การหาพารามิเตอร์ที่ดีที่สุดด้วย GridSearchCV:ในขั้นตอนนี้ ผมจะใช้ GridSearchCV เพื่อหาพารามิเตอร์ที่ดีที่สุดสำหรับโมเดล SVM และ KNN โดยเราจะทดสอบหลาย ๆ ค่าพารามิเตอร์ เพื่อหาค่าที่ให้ผลลัพธ์ดีที่สุด**"""

from sklearn.model_selection import GridSearchCV

# กำหนดพารามิเตอร์ที่ต้องการปรับ
param_grid_svm = {
    'C': [0.1, 1, 10],
    'kernel': ['linear', 'rbf'],
    'gamma': ['scale', 'auto']
}

# สร้างโมเดล SVM
svm = SVC()

# ใช้ GridSearchCV เพื่อหาค่าพารามิเตอร์ที่ดีที่สุด
grid_search_svm = GridSearchCV(svm, param_grid_svm, cv=5)
grid_search_svm.fit(X_train, y_train)

# แสดงผลลัพธ์ที่ดีที่สุด
print("Best Parameters for SVM:", grid_search_svm.best_params_)

from sklearn.neighbors import KNeighborsClassifier
param_grid_knn = {
    'n_neighbors': [3, 5, 7, 9],
    'weights': ['uniform', 'distance'],
    'metric': ['euclidean', 'manhattan']
}

# สร้างโมเดล KNN
knn = KNeighborsClassifier()

# ใช้ GridSearchCV เพื่อหาค่าพารามิเตอร์ที่ดีที่สุด
grid_search_knn = GridSearchCV(knn, param_grid_knn, cv=5)
grid_search_knn.fit(X_train, y_train)

# แสดงผลลัพธ์ที่ดีที่สุด
print("Best Parameters for KNN:", grid_search_knn.best_params_)

"""## **9. การฝึกโมเดล RandomForest:หลังจากฝึกโมเดล SVM แล้ว ผมจะสร้างโมเดล RandomForest ซึ่งเป็นโมเดลที่ใช้หลาย ๆ ต้นไม้ (trees) ในการตัดสินใจ ทำนายผลและประเมินผลลัพธ์**"""

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

# สร้างโมเดล RandomForest พร้อมพารามิเตอร์ class_weight='balanced'
rf = RandomForestClassifier(class_weight='balanced')

# กำหนดพารามิเตอร์ที่ต้องการปรับแต่ง
param_grid_rf = {
    'n_estimators': [50, 100],
    'max_depth': [None, 10],
    'min_samples_split': [2, 5]
}

# ใช้ GridSearchCV เพื่อหาค่าพารามิเตอร์ที่ดีที่สุด
grid_search_rf = GridSearchCV(rf, param_grid_rf, cv=3, n_jobs=-1)
grid_search_rf.fit(X_train, y_train)

# แสดงผลลัพธ์ที่ดีที่สุด
print("Best Parameters for RandomForest:", grid_search_rf.best_params_)

# ทำนายผล
y_pred_rf = grid_search_rf.predict(X_test)

# แสดงผลการทำนาย
print("Accuracy:", accuracy_score(y_test, y_pred_rf))
print("Classification Report:\n", classification_report(y_test, y_pred_rf))

"""## **10. เปรียบเทียบผลลัพธ์ของโมเดลทั้งหมด:สุดท้าย ผมจะเปรียบเทียบผลลัพธ์จากโมเดลที่ดีที่สุดจากทั้ง SVM, KNN, และ RandomForest เพื่อตัดสินใจว่าโมเดลไหนทำงานได้ดีที่สุด**"""

# ใช้โมเดลที่ได้จาก GridSearchCV และทำนายผล
best_svm = grid_search_svm.best_estimator_
best_knn = grid_search_knn.best_estimator_
best_rf = grid_search_rf.best_estimator_

# ประเมินผลของโมเดลที่ดีที่สุด
models = {
    'Best SVM': best_svm,
    'Best KNN': best_knn,
    'Best RandomForest': best_rf
}

# ประเมินผลและแสดงผล
# ประเมินผลและแสดงผล
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print(f"Model: {name}")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Classification Report:\n", classification_report(y_test, y_pred, zero_division=1))  # แก้ไขที่นี่
    print("="*50)

from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report

# สมมุติว่า X และ y คือ features และ target ของคุณ
X = data.drop('Loan_Status', axis=1)
y = data['Loan_Status']

# แบ่งข้อมูลเป็นชุดฝึกและชุดทดสอบ
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# สเกลข้อมูล (หากจำเป็น)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# สร้างโมเดล SVM และฝึกมัน
svm = SVC()
svm.fit(X_train, y_train)  # ฝึกโมเดล

# ทำนายผลด้วยชุดข้อมูลทดสอบ
y_pred = svm.predict(X_test)

# ประเมินผล
print("Model Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# ตรวจสอบว่าโมเดลใช้ kernel linear หรือไม่
if svm.kernel == 'linear':
    print("SVM Coefficients:", svm.coef_)

# ทดสอบโมเดลบนชุดข้อมูลทดสอบ
y_pred = svm.predict(X_test)
print("Model Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
import ipywidgets as widgets
from IPython.display import display

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

# สร้าง widget สำหรับรับข้อมูลจากผู้ใช้
gender_widget = widgets.Dropdown(options=['Male', 'Female'], description="Gender:")
marital_status_widget = widgets.Dropdown(options=['Yes', 'No'], description="Marital Status:")
education_widget = widgets.Dropdown(options=['Graduate', 'Not Graduate'], description="Education Level:")
dependents_widget = widgets.Dropdown(options=['0', '1', '2', '3+'], description="Dependents:")
income_widget = widgets.FloatText(description="Income:", value=0)
loan_amount_widget = widgets.FloatText(description="Loan Amount:", value=0)

# ปุ่ม Predict
predict_button = widgets.Button(description="Predict")

# ฟังก์ชันรับข้อมูลจากผู้ใช้
def get_user_input(gender, marital_status, education, dependents, income, loan_amount):
    try:
        gender = label_encoders['Gender'].transform([gender])[0]
    except ValueError:
        print(f"Invalid value for Gender. Defaulting to 'Unknown'.")
        gender = label_encoders['Gender'].transform(['Unknown'])[0]

    try:
        marital_status = label_encoders['Married'].transform([marital_status])[0]
    except ValueError:
        print(f"Invalid value for Marital Status. Defaulting to 'Unknown'.")
        marital_status = label_encoders['Married'].transform(['Unknown'])[0]

    # จัดการค่า Dependents ให้เป็น 3 ถ้าเป็น '3+'
    dependents = '3' if dependents == '3+' else dependents
    try:
        dependents = label_encoders['Dependents'].transform([str(dependents)])[0]
    except ValueError:
        print(f"Invalid value for Dependents. Defaulting to '0'.")
        dependents = label_encoders['Dependents'].transform(['0'])[0]

    try:
        education = label_encoders['Education'].transform([education])[0]
    except ValueError:
        print(f"Invalid value for Education. Defaulting to 'Unknown'.")
        education = label_encoders['Education'].transform(['Unknown'])[0]

    # สร้าง DataFrame สำหรับการทำนาย
    user_input = pd.DataFrame([[gender, marital_status, dependents, education, income, loan_amount]],
                              columns=['Gender', 'Married', 'Dependents', 'Education', 'Income', 'Loan_Amount'])

    # ปรับขนาดข้อมูลให้เหมือนกับข้อมูลที่ใช้ฝึก
    user_input_scaled = scaler.transform(user_input)
    return user_input_scaled

# ฟังก์ชันที่ทำงานเมื่อกดปุ่ม Predict
def on_predict_button_click(b):
    user_data = get_user_input(gender_widget.value, marital_status_widget.value,
                               education_widget.value, dependents_widget.value,
                               income_widget.value, loan_amount_widget.value)

    # ใช้โมเดล SVM เท่านั้น
    prediction = svm.predict(user_data)

    # แสดงผลการทำนาย
    if prediction[0] == 0:
        print("The loan will be denied.")
    else:
        print("The loan will be approved.")

# การเชื่อมต่อปุ่ม Predict กับฟังก์ชัน
predict_button.on_click(on_predict_button_click)

# การแสดง widget และปุ่ม
display(gender_widget, marital_status_widget, education_widget,
        dependents_widget, income_widget, loan_amount_widget, predict_button)