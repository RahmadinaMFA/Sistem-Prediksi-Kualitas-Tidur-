import pickle
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder


# Load model
model = pickle.load(open('tidur.sav', 'rb'))

# Judul Web
st.title('Prediksi Tidur Sehat')

# Fungsi untuk halaman Deskripsi
def show_deskripsi():
    st.write("Selamat datang di aplikasi Prediksi Tidur Sehat.")
    st.write("""
        Aplikasi ini dirancang untuk membantu Anda mengetahui kualitas tidur berdasarkan 
        data kesehatan seperti durasi tidur, tingkat stres, aktivitas fisik, dan lainnya. 
        Dengan teknologi Machine Learning, aplikasi ini memberikan prediksi apakah tidur 
        Anda termasuk kategori normal, insomnia, atau sleep apnea.
    """)
    st.write("Dataset : https://www.kaggle.com/code/tanshihjen/eda-sleep-health-and-lifestyle-dataset")

# Fungsi untuk halaman Dataset (Contoh dataset dapat ditambahkan)
def show_dataset():
    st.header("Dataset")
    st.write("Halaman ini menampilkan contoh dataset yang digunakan untuk melatih model.")
    # Contoh dataset
    data = {
        'Umur': [25, 35, 45],
        'Durasi Tidur': [7, 5, 6],
        'Kualitas Tidur': [8, 6, 7],
        'Aktivitas Fisik': [30, 20, 40],
        'Tingkat Stres': [5, 7, 6],
        'Denyut Jantung': [70, 80, 75],
        'Langkah Harian': [5000, 3000, 6000],
        'Pekerjaan': ['Engineer', 'Doctor', 'Teacher'],
        'BMI': ['Normal', 'Overweight', 'Normal Weight'],
        'Jenis Kelamin': ['Laki-Laki', 'Perempuan', 'Laki-Laki'],
        'Tekanan Darah (Sistolik)': [120, 130, 125],
        'Tekanan Darah (Diastolik)': [80, 85, 82]
    }
    df = pd.DataFrame(data)
    st.dataframe(df)

# Fungsi untuk halaman Grafik (Dummy grafik dapat diganti)
def show_grafik():
    st.header("Grafik Visualisasi Data")
    st.write("Halaman ini menampilkan beberapa grafik berdasarkan dataset.")

    # Contoh data
    data = {
        'Gender': ['Male', 'Female', 'Female', 'Male', 'Male'],
        'Occupation': ['Engineer', 'Doctor', 'Teacher', 'Engineer', 'Doctor'],
        'BMI Category': ['Normal', 'Overweight', 'Underweight', 'Normal', 'Obese'],
        'Sleep Disorder': ['Yes', 'No', 'Yes', 'No', 'Yes']
    }
    df = pd.DataFrame(data)

    # Label encoding
    label_encoder = LabelEncoder()
    df['gender'] = label_encoder.fit_transform(df['Gender'])
    df['Occ'] = label_encoder.fit_transform(df['Occupation'])
    df['BMI'] = label_encoder.fit_transform(df['BMI Category'])
    df['Sleep'] = label_encoder.fit_transform(df['Sleep Disorder'])

    # Grafik 1: Histogram Sleep Disorder
    st.subheader("Histogram Sleep Disorder")
    fig1 = plt.figure(figsize=(3, 3))
    sns.set(font_scale=0.8)
    sns.histplot(data=df, x='Sleep Disorder')
    st.pyplot(fig1)

    # Grafik 2: Distribution of BMI Categories
    st.subheader("Distribusi Kategori BMI")
    bmi_counts = df['BMI Category'].value_counts()
    fig2 = plt.figure(figsize=(8, 6))
    plt.bar(bmi_counts.index, bmi_counts.values)
    plt.title("Distribution of BMI Categories")
    plt.xlabel("BMI Category")
    plt.ylabel("Count")
    plt.xticks(rotation=45)
    st.pyplot(fig2)

    # Grafik 3: Distribusi Pekerjaan
    st.subheader("Distribusi Pekerjaan")
    occ_counts = df['Occupation'].value_counts()
    fig3 = plt.figure(figsize=(8, 3))
    plt.bar(occ_counts.index, occ_counts.values)
    plt.title("Distribusi dari Pekerjaan")
    plt.xlabel("Pekerjaan")
    plt.ylabel("Count")
    plt.xticks(rotation=45)
    st.pyplot(fig3)

    st.write("Data setelah encoding:")
    st.write(df.head())

# Fungsi untuk halaman Prediksi
def show_prediksi():
    st.header("Prediksi Tidur Sehat")
    st.write("Masukkan informasi pasien untuk memprediksi kualitas tidur:")

    # Input data pengguna
    Age = st.number_input('Umur Pasien', min_value=0, max_value=120, value=25)
    Sleep_Duration = st.number_input('Berapa jam pasien tidur setiap harinya?', min_value=0.0, max_value=24.0, value=7.0)
    Quality_of_Sleep = st.number_input('Dari 1 - 10, Berapa kualitas tidur pasien?', min_value=1, max_value=10, value=8)
    Physical_Activity_Level = st.number_input('Total berapa menit pasien melakukan aktivitas fisik per hari?', value=30)
    Stress_Level = st.number_input('Dari 1 - 10, Berapa tingkat stres pasien?', min_value=1, max_value=10, value=5)
    Heart_Rate = st.number_input('Denyut jantung istirahat pasien per menit', value=70)
    Daily_Steps = st.number_input('Jumlah langkah harian pasien', value=5000)
    Occ = st.selectbox('Pekerjaan pasien', ['Nurse', 'Doctor', 'Engineer', 'Lawyer', 'Teacher'])
    BMI = st.selectbox('Kategori BMI pasien', ['Normal', 'Overweight', 'Normal Weight', 'Obese'])
    gender = st.selectbox('Jenis Kelamin Pasien', ['Laki-Laki', 'Perempuan'])
    BPU = st.number_input('Tekanan Darah Sistolik Pasien', value=120)
    BPD = st.number_input('Tekanan Darah Diastolik Pasien', value=80)

    # Encode data
    from sklearn.preprocessing import LabelEncoder
    label_encoder_occ = LabelEncoder()
    label_encoder_occ.fit(['Nurse', 'Doctor', 'Engineer', 'Lawyer', 'Teacher'])
    encoded = label_encoder_occ.transform([Occ])[0]

    label_encoder_bmi = LabelEncoder()
    label_encoder_bmi.fit(['Normal', 'Overweight', 'Normal Weight', 'Obese'])
    bmi_encoded = label_encoder_bmi.transform([BMI])[0]

    label_encoder_gender = LabelEncoder()
    label_encoder_gender.fit(['Laki-Laki', 'Perempuan'])
    gender_encoded = label_encoder_gender.transform([gender])[0]

    # Prediksi
    if st.button('Proses'):
        predict = model.predict(
            [[Age, Sleep_Duration, Quality_of_Sleep, Physical_Activity_Level, 
              Stress_Level, Heart_Rate, Daily_Steps, encoded, bmi_encoded, gender_encoded, BPU, BPD]]
        )
        if predict == 0:
            st.write('Kualitas tidur pasien termasuk dalam kategori **Insomnia**.')
        elif predict == 1:
            st.write('Kualitas tidur pasien termasuk dalam kategori **Normal**.')
        else:
            st.write('Kualitas tidur pasien termasuk dalam kategori **Sleep Apnea**.')

# Sidebar untuk navigasi halaman
add_selectbox = st.sidebar.selectbox(
    "PILIH MENU",
    ("Deskripsi", "Dataset", "Grafik", "Prediksi")
)

# Menjalankan fungsi berdasarkan pilihan
if add_selectbox == "Deskripsi":
    show_deskripsi()
elif add_selectbox == "Dataset":
    show_dataset()
elif add_selectbox == "Grafik":
    show_grafik()
elif add_selectbox == "Prediksi":
    show_prediksi()
