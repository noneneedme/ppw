import streamlit as st
import requests
from bs4 import BeautifulSoup
import csv
import pandas as pd
import base64  # Import modul base64
import re
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

# Mengimpor modul data_functions.py

def bersih(data):
    # Pindahkan pemrosesan data dari kode pertama ke sini
    data = data.dropna(subset=['abstrak'])
    data = data.reset_index(drop=True)
    data['abstrak'] = data['abstrak'].str.lower()
    data_lower_case = data['abstrak']

    clean = []

    for i in range(len(data['abstrak'])):
        clean_symbols = re.sub("[^a-zA-Zï ]+", " ", data['abstrak'].iloc[i])  # Pembersihan karakter
        clean_tag = re.sub("@[A-Za-z0-9_]+", "", clean_symbols)  # Pembersihan mention
        clean_hashtag = re.sub("#[A-Za-z0-9_]+", "", clean_tag)  # Pembersihan hashtag
        clean_https = re.sub(r'http\S+', '', clean_hashtag)  # Pembersihan URL link
        clean_whitespace = re.sub(r'\s+', ' ', clean_https).strip()  # Mengganti spasi berlebih dengan spasi tunggal
        clean.append(clean_whitespace)

    clean_result = pd.DataFrame(clean, columns=['Cleansing Abstrak'])
    return clean_result, clean

def token(clean):
    # Slank Word
    import nltk
    nltk.download('stopwords')
    nltk.download('punkt')
    from nltk.tokenize import sent_tokenize, word_tokenize
    from nltk.corpus import stopwords
    # Membuat kamus slang words dan kata Indonesia yang benar
    slang_dict = pd.read_csv("https://raw.githubusercontent.com/noneneedme/ppw/main/combined_slang_words.txt", sep=" ", header=None)

    # Membuat fungsi untuk mengubah slang words menjadi kata Indonesia yang benar
    def replace_slang_words(text):
        words = nltk.word_tokenize(text.lower())
        words_filtered = [word for word in words if word not in stopwords.words('indonesian')]
        for i in range(len(words_filtered)):
            if words_filtered[i] in slang_dict:
                words_filtered[i] = slang_dict[words_filtered[i]]
        return ' '.join(words_filtered)

    # Contoh penggunaan
    slang_words=[]
    for i in range(len(clean)):
        slang = replace_slang_words(clean[i])
        slang_words.append(slang)

    data_slang = pd.DataFrame(slang_words, columns=["Slang Word Corection"])

    words = []
    for i in range (len(data_slang)):
        tokens = word_tokenize(slang_words[i])
        listStopword =  set(stopwords.words('indonesian'))

        removed = []
        for t in tokens:
            if t not in listStopword:
                removed.append(t)

        words.append(removed)
    return removed, words

def topikdok(words, judul):
    gabung=[]
    for i in range(len(words)):
        joinkata = ' '.join(words[i])
        gabung.append(joinkata)

    result = pd.DataFrame(gabung, columns=['Join_Kata'])
    result = result.dropna(subset=['Join_Kata'])
    # Extract the 'Join_Kata' column
    gabung = result['Join_Kata'].tolist()

    # TfidfVectorizer
    tfidfvectorizer = TfidfVectorizer(analyzer='word', stop_words='english')
    tfidf_wm = tfidfvectorizer.fit_transform(gabung)
    tfidf_tokens = tfidfvectorizer.get_feature_names_out()
    df_tfidfvect = pd.DataFrame(data=tfidf_wm.toarray(), columns=tfidf_tokens)

    # CountVectorizer
    countvectorizer = CountVectorizer(analyzer='word', stop_words='english')
    count_wm = countvectorizer.fit_transform(gabung)
    count_tokens = countvectorizer.get_feature_names_out()
    df_countvect = pd.DataFrame(data=count_wm.toarray(), columns=count_tokens)

    lda = LatentDirichletAllocation(n_components=3, doc_topic_prior=0.2, topic_word_prior=0.1,random_state=42,max_iter=1)
    lda_top=lda.fit_transform(df_countvect)
    topics = pd.DataFrame(lda_top, columns=['Topik 1','Topik 2','Topik 3'])

    satukan = pd.concat([judul, topics], axis=1)
    return satukan


# Fungsi untuk menampilkan data
def display_github_csv_data(link):
    data = pd.read_csv(link)
    return data
    

def display_uploaded_csv_data(uploaded_file):
    data = pd.read_csv(uploaded_file)
    return data


def add_trailing_slash(url):
    if not url.endswith("/"):
        return url + "/"
    return url

# Fungsi untuk scraping data PTA Trunojoyo
def scraping_pta(url, batas):
    st.title("Scraper Data PTA Trunojoyo")
    st.subheader("Hasil Scraping:")
    st.write("Data scraping akan ditampilkan di sini.")

    data = []
    urldef = 'https://pta.trunojoyo.ac.id/c_search/byprod/7/'

    for page in range(1, batas+1):
        if url == "":
            url = urldef
        req = requests.get(url + str(page))
        soup = BeautifulSoup(req.text, 'html.parser')
        items = soup.findAll('li', {'data-cat': '#luxury'})

        for item in items:
            link = item.find('a', 'gray button')['href']
            print(link)

            req2 = requests.get(link)
            soup2 = BeautifulSoup(req2.text, 'html.parser')

            penulis_elem = soup2.find('div', {'style': 'padding:2px 2px 2px 2px;'}).find('span')
            penulis = penulis_elem.text
            print(penulis)

            dospem_elem = soup2.find('div', {'style': 'float:left; width:540px;'}).findAll('div', {'style': 'padding:2px 2px 2px 2px;'})

            dospem_i = 'Dosen Pembimbing I tidak ditemukan'
            dospem_ii = 'Dosen Pembimbing II tidak ditemukan'

            for dospem in dospem_elem:
                dospem_text = dospem.find('span').text
                if 'Dosen Pembimbing I :' in dospem_text:
                    dospem_i = dospem_text.replace('Dosen Pembimbing I :', '').strip()
                elif 'Dosen Pembimbing II :' in dospem_text:
                    dospem_ii = dospem_text.replace('Dosen Pembimbing II :', '').strip()
            print("Dosen Pembimbing I :", dospem_i, "\nDosen Pembimbing II :", dospem_ii)

            judul_elem = item.find('a', 'title')
            judul = judul_elem.text
            print(judul)

            absk_elem = soup2.find('div', {'style': 'margin: 15px 15px 15px 15px;'}).find('p')
            absk = absk_elem.text if absk_elem else 'Abstrak tidak ditemukan'
            print(absk)

            data.append([judul, penulis, dospem_i, dospem_ii, absk])

    return data

def tfidf_calculation():
    import pandas as pd
    from sklearn.feature_extraction.text import TfidfVectorizer
    st.title("Topik Modeling")
    
    data = None  # Initialize 'data' with None
    
    # Pilihan antara input link atau file CSV
    opsi_preprocessing = st.radio("Pilih Opsi:", ["Input Link Github", "Upload File CSV"])

    if opsi_preprocessing == "Input Link Github":
        link = st.text_input("Masukkan Link URL Github:", "https://raw.githubusercontent.com/noneneedme/ppw/main/dataptaman%20(1).csv")
        if st.button("Tampilkan Data"):
            data = display_github_csv_data(link)
            if data is not None:
                st.subheader("Data:")
                st.write(data)  # Tampilkan data pada sidebar
        if st.button("Jalankan"):
            data = display_github_csv_data(link)
            if data is not None:  # Check if 'data' is assigned
                judul = data['Judul']
                clean_result, clean = bersih(data)
                removed, words = token(clean)
                satukan=topikdok(words, judul)
                st.success("Proses Preprocessing Selesai.")
                st.success("Data cleaning")
                st.dataframe(clean_result)
                st.success("LDA Judul to Topik")
                st.dataframe(satukan)
        
    elif opsi_preprocessing == "Upload File CSV":
        uploaded_file = st.file_uploader("Upload File CSV:", type=["csv"])
        if uploaded_file is not None:
            data = display_uploaded_csv_data(uploaded_file)
            if st.button("Tampilkan Data"):
                if data is not None:
                    st.subheader("Data:")
                    st.write(data)  # Tampilkan data pada sidebar
            if st.button("Jalankan"):
                data = display_uploaded_csv_data(uploaded_file)
                if data is not None:  # Check if 'data' is assigned
                    judul = data['Judul']
                    clean_result, clean = bersih(data)
                    removed, words = token(clean)
                    satukan=topikdok(words, judul)
                    st.success("Proses Preprocessing Selesai.")
                    st.success("Data cleaning")
                    st.dataframe(clean_result)
                    st.success("LDA Judul to Topik")
                    st.dataframe(satukan)

option = st.sidebar.selectbox("Pilih Opsi:", ["Scraping Data PTA", "Topik Modeling"])

if option == "Scraping Data PTA":
    url = st.text_input("Masukkan URL PTA Contoh:", "https://pta.trunojoyo.ac.id/c_search/byprod/7/")
    batas = st.number_input("Masukkan halaman akhir yang akan di scraping/crawling:", min_value=1, max_value=207, value=207)  # Nilai default adalah 207
    
    # Tampilkan tombol "Cek Ketersediaan URL"
    if st.button("Cek Ketersediaan URL"):
        # Kirim permintaan GET ke URL
        response = requests.get(url)
        
        # Periksa status kode respons
        if response.status_code == 200:
            # Jika URL tersedia, tampilkan tombol "Mulai Scraping"
            if st.button("Mulai Scraping"):
                data = scraping_pta(url, batas)
                
                # Form input untuk nama file CSV
                nama_file = st.text_input("Nama File CSV:", "dataptaman.csv")
                
                # Tampilkan tombol "Simpan Data"
                if st.button("Simpan Data"):
                    pta = pd.DataFrame(data, columns=['Judul', 'penulis', 'Dosen Pembimbing I', 'Dosen Pembimbing II', 'abstrak'])
                    csv_data = pta.to_csv(index=False)

                    # Buat tautan unduhan untuk file CSV
                    b64 = base64.b64encode(csv_data.encode()).decode()
                    href = f'<a href="data:file/csv;base64,{b64}" download="{nama_file}">Klik di sini untuk mengunduh {nama_file}</a>'
                    st.markdown(href, unsafe_allow_html=True)
                    
                    st.success(f'Data berhasil disimpan ke {nama_file}')
        else:
            st.error('URL tidak tersedia. Status kode:', response.status_code)
else:
    tfidf_calculation()


