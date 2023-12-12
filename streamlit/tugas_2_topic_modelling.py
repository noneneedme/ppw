import streamlit as st
import requests as req
from bs4 import BeautifulSoup as bs
import pandas as pd
import csv
import re
import numpy as np
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
import nltk
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.cluster import KMeans
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.metrics import silhouette_score
nltk.download('stopwords')
nltk.download('punkt')
import validators
server= {"user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/110.0.5481.104 Safari/537.36"}
import base64

def scrape_detik (hal):
  global server
  data=[]
  for page in range (1, hal+1):
    #st.write(page)
    url = f"https://www.detik.com/search/searchall?query=pemilu+2024&siteid={page}"
    request = req.get(url,server).text
    # memanggil beautysoup
    soup = bs(request, "lxml")
    #menampilkan list berita
    daftar_berita = soup.find("div","list media_rows list-berita")
    # menampil isi artikel
    artikel = daftar_berita.find_all("article")

    for tampil in artikel:
      #judul
      judul = tampil.find_all("h2")
      judul = [ele.text.strip() for ele in judul]
      # deskripsi
      isi  = tampil.find_all("p")
      isi  = [ele.text.strip() for ele in isi]
      data.append([judul[0], isi[0]])
  return data

def dat(url):
    data = pd.read_csv(url)  # Example using pandas to read CSV data
    return data
  
def run():
    st.title("Tugas 2 Topik Modeling")
    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9, tab10, tab11, tab12, tab13, tab14 = st.tabs(
        ["Crawling", "Data", "Cleansing", "Slank Word", "Kostum Stopword",
         "Join Kata", "TF-IDF Vect", "TF-IDF Clus", "Tabel LDA",
         "Judul-Topik", "Klastering", "K-Means", "Klastering", "Filtrasi"]
        )
    with tab1:
      st.title('crawling/scraping')
      berita=scrape_detik(200)
      isi_berita = pd.DataFrame(berita)
      isi_berita.columns=["Judul","Isi"]
      st.write(isi_berita)

      isi_berita.to_csv("detik.csv" , index=False)
      st.markdown("### [Download file CSV](data:file/csv;base64,{})".format(base64.b64encode(isi_berita.to_csv(index=False).encode()).decode()), unsafe_allow_html=True)

    with tab2:
      st.title('CSV Data Viewer')

      user_input = st.text_input("Masukkan URL", "https://raw.githubusercontent.com/Mohammad-Unis-Iswahyudi/ppw/main/Data-PTA-Teknik-Industri.csv")

      if user_input != "https://":
          if validators.url(user_input):
            if st.button('Gunakan'):
              st.write("URL valid:", user_input)
              data = dat(user_input)
          else:
              data = isi_berita
      data = data.dropna(subset=['Isi'])
      data = data.reset_index(drop=True)
      st.write(data)

      #data['abstrak'].fillna('', inplace=True)
      st.write('Jumlah data')
      jumlah_entri = data.shape[0]
      st.write(jumlah_entri)

      st.write('Isi')
      st.write(data['Isi'])
      st.write('Judul')
      st.write(data['Judul'])

      data['Isi'] = data['Isi'].str.lower()
      data_lower_case = data['Isi']
      st.write('Lower case isi')
      st.write(data_lower_case)

    with tab3:
      st.title('Cleansing')
      clean = []

      for i in range(len(data['Isi'])):
          clean_symbols = re.sub("[^a-zA-Zï ]+", " ", data['Isi'].iloc[i])  # Pembersihan karakter
          clean_tag = re.sub("@[A-Za-z0-9_]+", "", clean_symbols)  # Pembersihan mention
          clean_hashtag = re.sub("#[A-Za-z0-9_]+", "", clean_tag)  # Pembersihan hashtag
          clean_https = re.sub(r'http\S+', '', clean_hashtag)  # Pembersihan URL link
          clean_whitespace = re.sub(r'\s+', ' ', clean_https).strip()  # Mengganti spasi berlebih dengan spasi tunggal
          clean.append(clean_whitespace)


      clean_result = pd.DataFrame(clean,columns=['Cleansing Isi'])
      st.write(clean_result)

    with tab4:
      st.title('Slank Word')
      slang_dict = pd.read_csv("https://raw.githubusercontent.com/Mohammad-Unis-Iswahyudi/prosaindata/main/combined_slang_words.txt", sep=" ", header=None)

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
      st.write(data_slang)

    with tab5:
      st.title('Kostum Stopword')
      # Daftar kata yang ingin ditambahkan sebagai stopword
      custom_stopwords = ['ab', 'zam', 'abai', 'aadalah', 'abdoel', 'ppp']

      words = []
      for i in range (len(data_slang)):
        tokens = word_tokenize(slang_words[i])
        listStopword =  set(stopwords.words('indonesian'))

        # Menambahkan kata-kata custom ke dalam set stopword
        listStopword.update(custom_stopwords)

        removed = []
        for t in tokens:
            if t not in listStopword:
                removed.append(t)

        words.append(removed)
        st.write(removed)

    with tab6:
      st.title('Join Kata')
      gabung=[]
      for i in range(len(words)):
        joinkata = ' '.join(words[i])
        gabung.append(joinkata)

      result = pd.DataFrame(gabung, columns=['Join_Kata'])
      st.write(result)

    with tab7:
      st.title('TF-IDF Vector')
      # Load the CSV file
      df = pd.read_csv('https://raw.githubusercontent.com/Mohammad-Unis-Iswahyudi/ppw/main/detikpro.csv')

      # Check for and remove rows with missing values in the 'Join_Kata' column
      df = df.dropna(subset=['Join_Kata'])

      # Extract the 'Join_Kata' column
      gabung = df['Join_Kata'].tolist()

      countvectorizer = CountVectorizer(analyzer= 'word', stop_words='english')
      tfidfvectorizer = TfidfVectorizer(analyzer='word',stop_words= 'english')
      count_wm = countvectorizer.fit_transform(gabung)
      tfidf_wm = tfidfvectorizer.fit_transform(gabung)

      count_tokens = countvectorizer.get_feature_names_out()
      tfidf_tokens = tfidfvectorizer.get_feature_names_out()
      df_countvect = pd.DataFrame(data = count_wm.toarray(),columns = count_tokens)
      df_tfidfvect = pd.DataFrame(data = tfidf_wm.toarray(),columns = tfidf_tokens)
      df_countvect['Judul'] = data["Judul"]
      columns = ['Judul'] + [col for col in df_countvect.columns if col != 'Judul']
      df_countvect = df_countvect[columns]
      st.write("Count Vectorizer\n")
      st.write(df_countvect)

    with tab8:
      st.title('Klastering TF-IDF')
      # Jumlah cluster yang diinginkan
      num_clusters = 2

      # Menginisialisasi model K-Means
      kmeans = KMeans(n_clusters=num_clusters, random_state=0)

      # Melakukan clustering pada data TF-IDF
      clusters = kmeans.fit_predict(tfidf_wm)

      # Menambahkan kolom cluster ke DataFrame
      df_countvect['Cluster'] = clusters

      # Tampilkan hasil
      df_countvect[['Judul', 'Cluster']]

      # Hapus kolom 'Judul' dari DataFrame df_countvect
      df_countvect = df_countvect.drop(columns=['Judul'])

      # Tampilkan DataFrame tanpa kolom 'Judul'
      st.write("Count Vectorizer\n")
      df_countvect

    with tab9:
      st.title('LDA')
      lda = LatentDirichletAllocation(n_components=2, doc_topic_prior=0.2, topic_word_prior=0.1,random_state=42,max_iter=1)
      lda_top=lda.fit_transform(df_countvect)

      st.write(lda_top.shape)
      st.write(lda_top)
      st.write(lda.components_)
      st.write(lda.components_.shape)  

    with tab10:
      st.title('Judul-Topik')
      st.write('Topik')
      topics = pd.DataFrame(lda_top, columns=['Topik 1','Topik 2'])
      st.write(topics)

      st.write('Judul')
      judul=data['Judul']
      st.write(judul)

      st.write('Gabungan')
      gabung = pd.concat([judul, topics], axis=1)
      st.write(gabung)

    with tab11:
      st.title('Klastering')
      label=[]
      for i in range (1,(lda.components_.shape[1]+1)):
        masukan = df_countvect.columns[i-1]
        label.append(masukan)
      VT_tabel = pd.DataFrame(lda.components_,columns=label)
      VT_tabel.rename(index={0:"Topik 1",1:"Topik 2",2:"Topik 3"}).transpose()

      st.write(VT_tabel)

    with tab12:
      st.title('K-Means')
      kmeans = KMeans(n_clusters=2, random_state=42)
      kmeans.fit(lda_top)
      cluster_labels = kmeans.labels_
      data = {'Dokumen': range(len(cluster_labels)), 'Cluster': cluster_labels}
      duf = pd.DataFrame(data)
      st.write(duf)

    with tab13:
      st.title('Klasterisasi')
      # KMeans clustering
      kmeans = KMeans(n_clusters=2, random_state=42)
      kmeans.fit(lda_top)
      cluster_labels = kmeans.labels_

      # Hitung Silhouette Coefficient
      silhouette_avg = silhouette_score(lda_top, cluster_labels)
      st.write("Silhouette Coefficient:", silhouette_avg)

    with tab14:
      # Filter dokumen dengan cluster 0 dan 1
      cluster_0_documents = duf[duf['Cluster'] == 0]
      cluster_1_documents = duf[duf['Cluster'] == 1]

      # Ambil judul dokumen untuk masing-masing cluster
      cluster_0_document_titles = judul[cluster_0_documents['Dokumen']]
      cluster_1_document_titles = judul[cluster_1_documents['Dokumen']]

      # Buat DataFrame untuk masing-masing cluster
      cluster_0_df = pd.DataFrame({'Judul Dokumen': cluster_0_document_titles, 'Dokumen': cluster_0_documents['Dokumen'], 'Cluster': cluster_0_documents['Cluster']})
      cluster_1_df = pd.DataFrame({'Judul Dokumen': cluster_1_document_titles, 'Dokumen': cluster_1_documents['Dokumen'], 'Cluster': cluster_1_documents['Cluster']})

      # Tampilkan tabel untuk Cluster 0
      st.write("Dokumen Cluster 0:")
      st.write(cluster_0_df)

      # Tampilkan tabel untuk Cluster 1
      st.write("\nDokumen Cluster 1:")
      st.write(cluster_1_df)
