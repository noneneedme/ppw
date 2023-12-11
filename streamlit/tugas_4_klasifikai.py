import requests
import streamlit as st
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
import pandas as pd
import re
import validators
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier

def scrap(data_list, category, start_page=2):
    page_number = start_page

    while len(data_list) < 366:
        url = f"https://www.antaranews.com/{category}/{page_number}"
        req = requests.get(url)
        soup = BeautifulSoup(req.text, 'html.parser')
        items = soup.findAll('article', {'class': 'simple-post simple-big clearfix'})

        for item in items:
            link = item.find('a')['href']
            response = requests.get(link)
            html = response.text

            soup = BeautifulSoup(html, 'html.parser')
            judul_elem = soup.find('h1', class_="post-title")
            article = soup.find('div', class_="post-content clearfix")
            if article:
                article_text = article.get_text()
                judul = judul_elem.text
                data_list.append([judul, article_text, category])

        page_number += 1  # Naikkan nomor halaman untuk mengambil halaman berikutnya

        if page_number > 100:  # Pastikan untuk menghentikan jika nomor halaman melebihi batas yang ada
            break

def run():
    st.title("Tugas 1 PTA")
    tab0, tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9, tab10, tab11, tab12, tab13, tab14 = st.tabs(
        ["Crawling", "Data", "Preproses", "Tokenisasi", "TF-IDF", "TF-IDF Kalimat", "Cleansing",
         "Vektorisasi", "PCA", "Split Data", "Sampling", "Model KNN", "Model MLP",
         "Model Naive Bayes", "Model Random Forest"])
    with tab0:
        if st.button('Mulai Scraping'):
            # Buat DataFrame untuk setiap kategori
            politik=[]
            ekonomi=[]
            olahraga=[]

            scrap(politik, 'politik')
            scrap(ekonomi, 'ekonomi')
            scrap(olahraga, 'olahraga')

            st.write(len(politik))
            st.write(len(ekonomi))
            st.write(len(olahraga))

            df_politik = pd.DataFrame(politik, columns=['judul', 'isi', 'label'])
            df_ekonomi = pd.DataFrame(ekonomi, columns=['judul', 'isi', 'label'])
            df_olahraga = pd.DataFrame(olahraga, columns=['judul', 'isi', 'label'])

            # Gabungkan DataFrame untuk setiap kategori menjadi satu DataFrame tunggal
            df_combined = pd.concat([df_politik, df_ekonomi, df_olahraga])

            # Simpan DataFrame ke dalam file CSV
            csv=df_combined.to_csv(index=False)

            st.write('Success')
            st.download_button(
                label="Download data as CSV",
                data=csv,
                file_name='gabungan_berita.csv',
                mime='text/csv',
            )
    with tab1:
        st.title('Read Data')
        uploaded_files = st.file_uploader("Choose a CSV file", accept_multiple_files=False)
        for uploaded_file in uploaded_files:
            df_combined = uploaded_file.read()
            st.write("filename:", uploaded_file.name)
            st.write(df_combined)
                
    with tab2:
        st.title('Preproses')
        # Melakukan pre-processing pada kolom 'isi' untuk menghilangkan karakter \t, \n, atau \r
        df_combined['isi'] = df_combined['isi'].replace(r'[\t\n\r]', ' ', regex=True)

        # Mengatasi spasi berlebih dalam kolom 'isi'
        df_combined['isi'] = df_combined['isi'].apply(lambda x: re.sub(r'\s+', ' ', x))
        df_combined

        nltk.download('punkt')

        # Ambil teks dari kolom 'isi' dalam DataFrame
        isi = df_combined['isi']

    with tab3:
        st.title('tokenisasi')
        # Tokenisasi teks menjadi kalimat
        for text in isi:
            sentences = nltk.sent_tokenize(text)
            # Cetak kalimat-kalimat
            for sentence in sentences:
                st.write(sentence)

        tfidf_vectorizer = TfidfVectorizer()

    with tab4:
        st.title('Menghitung TF-IDF')
        # Hitung TF-IDF
        tfidf_matrix = tfidf_vectorizer.fit_transform(sentences)
        # Daftar kata kunci
        feature_names = tfidf_vectorizer.get_feature_names_out()
        # Konversi matriks TF-IDF menjadi bentuk yang lebih mudah dibaca
        tfidf_values = tfidf_matrix.toarray()
        # Ambil daftar kata dari feature_names
        words = feature_names

    with tab5:
        st.title('Cetak TF-IDF')
        # Cetak TF-IDF untuk setiap kata dalam setiap kalimat
        for i, sentence in enumerate(sentences):
            st.write(f"Kalimat {i + 1}: {sentence}")
            for j, word in enumerate(words):
                tfidf_value = tfidf_values[i][j]
                if tfidf_value > 0:
                    st.write(f"{word}: {tfidf_value:.4f}")
            st.write()
    with tab6:
        st.title('clemsing')
        #Cleansing number from text
        sentences = [re.sub(r'\d+', '', sentence) for sentence in sentences]
        sentences = [re.sub("#[A-Za-z0-9_]+", "", sentence) for sentence in sentences]  # Cleansing hashtag
        sentences = [re.sub(r'http\S+', '', sentence) for sentence in sentences]  # Cleansing URL link
        sentences = [re.sub("[^a-zA-Zï ]+", " ", sentence) for sentence in sentences]  # Cleansing characters

    with tab7:
        st.title('Vektor TF-IDF')
        # Inisialisasi penghitung TF-IDF
        tfidf_vectorizer = TfidfVectorizer()

        # Hitung TF-IDF
        tfidf_matrix = tfidf_vectorizer.fit_transform(sentences)

        # Daftar kata kunci
        feature_names = tfidf_vectorizer.get_feature_names_out()

        # Konversi matriks TF-IDF menjadi bentuk yang lebih mudah dibaca
        tfidf_values = tfidf_matrix.toarray()

        # Membuat DataFrame untuk menyimpan data TF-IDF
        df_tfidf = pd.DataFrame(tfidf_values, columns=feature_names)

        # Menampilkan DataFrame
        st.write(df_tfidf)

    with tab8:
        st.title('PCA')
        # PCA (Reduksi Dimensi)
        pca_abstrak = PCA(n_components=15)
        principalComponents_abstrak = pca_abstrak.fit_transform(df_tfidf)
        principal_abstrak_Df = pd.DataFrame(principalComponents_abstrak)
        st.write(principal_abstrak_Df)

    with tab9:
        st.title('splitting Data')
        # Splitting Dta
        training, test = train_test_split(tfidf_values,test_size=0.2, random_state=42)#Nilai X training dan Nilai X testing
        training_label, test_label = train_test_split(df_combined['label'], test_size=0.2, random_state=1)#Nilai Y training dan Nilai Y testing

        st.write(training.shape)   # Check the shape of training features
        st.write(training_label.shape)


    with tab10:
        st.title('Sampling dan Splitting')
        # Ambil 16 sampel secara acak dari data fitur (X) dan sesuaikan dengan labelnya (Y)
        sampled_data = df_tfidf.sample(n=16, random_state=42)  # Mengambil 16 sampel secara acak dari df_tfidf
        X_sampled = sampled_data  # Data fitur yang diambil secara acak
        Y_sampled = df_combined.loc[sampled_data.index, 'label']  # Mengambil label yang sesuai

        # Bagi data menjadi training dan testing
        X_train, X_test, y_train, y_test = train_test_split(X_sampled, Y_sampled, test_size=0.2, random_state=42)

        st.write(X_train.shape)
        st.write(X_test.shape)
        st.write(y_train.shape)
        st.write(y_test.shape)

    with tab11:
        st.title('Model KNN')
        # Pemodelan KNN

        modelKNN = KNeighborsClassifier(n_neighbors=5)
        modelKNN.fit(X_train, y_train)

        test_pred = modelKNN.predict(X_test)
        st.write(test_pred)

        st.write(accuracy_score(y_test, test_pred))

        st.write(classification_report(y_test, test_pred))

    with tab12:
        st.title('Model MLP')
        # Inisialisasi model MLP
        model_mlp = MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=500)
        # hidden_layer_sizes: Jumlah neuron di setiap hidden layer
        # max_iter: Jumlah iterasi yang diizinkan untuk training

        # Melatih model dengan data training
        model_mlp.fit(X_train, y_train)

        # Melakukan prediksi dengan data testing
        predictions_mlp = model_mlp.predict(X_test)


        # Menghitung akurasi
        accuracy_mlp = accuracy_score(y_test, predictions_mlp)
        st.write(f"Akurasi model MLP: {accuracy_mlp:.2f}")

        # Menampilkan classification report
        st.write(classification_report(y_test, predictions_mlp))

    with tab13:
        st.title('Model Naive Bayes')
        # Inisialisasi model Naive Bayes (Gaussian Naive Bayes)
        model_nb = GaussianNB()

        # Melatih model dengan data training
        model_nb.fit(X_train, y_train)

        # Prediksi dengan data testing menggunakan model Naive Bayes
        predictions_nb = model_nb.predict(X_test)

        # Menghitung akurasi model Naive Bayes
        accuracy_nb = accuracy_score(y_test, predictions_nb)
        st.write(f"Akurasi model Naive Bayes: {accuracy_nb:.2f}")

    with tab14:
        st.title('Model Random Forest')
        # Inisialisasi model Random Forest
        model_rf = RandomForestClassifier(n_estimators=100, random_state=42)
        # Melatih model dengan data training
        model_rf.fit(X_train, y_train)
        # Prediksi dengan data testing menggunakan model Random Forest
        predictions_rf = model_rf.predict(X_test)
        # Menghitung akurasi model Random Forest
        accuracy_rf = accuracy_score(y_test, predictions_rf)
        st.write(f"Akurasi model Random Forest: {accuracy_rf:.2f}")

        st.write(accuracy_score(y_test, predictions_rf))
