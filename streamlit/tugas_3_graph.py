import nltk
import streamlit as st
nltk.download('stopwords')
nltk.download('punkt')
import requests
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import validators

def ngscrap():
    # Unduh konten halaman web berita
    url = "https://www.antaranews.com/berita/3821805/bmkg-prediksi-hujan-disertai-kilat-terjadi-di-jakarta-selatan?utm_source=antaranews&utm_medium=desktop&utm_campaign=terkini"
    response = requests.get(url)
    html = response.text

    # Parsing halaman web menggunakan BeautifulSoup
    soup = BeautifulSoup(html, 'html.parser')

    # Ekstraksi teks dari elemen-elemen yang berisi berita
    article = soup.find('div', class_="post-content clearfix")  # Sesuaikan dengan struktur HTML halaman web berita
    article_text = article.get_text()
    return article_text

# Preprocessing
# Lowercasing

def run():
    st.title("Tugas 1 PTA")
    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9, tab10, tab11, tab12, tab13 = st.tabs(
        ["Crawling", "Cleansing", "Tokenisasi", "TF-IDF", "Indexing", "Cos Sim",
         "Cos Sim Frame", "Grafik Matrik", "Grafik Sim", "Closeness Central",
         "Page Rank 1", "Page Rank 2", "Page Rank3"])

    with tab1:
        st.title('Scraping/Crawling')
        if st.button('mulai scrap/crawling'):
            article_text = ngscrap()
            article_text = article_text.lower()

    with tab2:
        st.title('Cleansing')
        # Cleaning
        article_text = ''.join(e for e in article_text if (e.isalnum() or e.isspace() or e == '.'))

        # Hapus Angka
        article_text = ''.join([char for char in article_text if not char.isdigit()])

    with tab3:
        st.title('Tokenisasi')
        # Tokenisasi teks menjadi kalimat menggunakan nltk
        sentences = nltk.sent_tokenize(article_text)

        # Tokenisasi setiap kalimat menjadi kata-kata
        words = [nltk.word_tokenize(sentence) for sentence in sentences]

        # Stopword Removal
        stop_words = set(stopwords.words('indonesian'))
        filtered_sentences = []


        for sentence in words:
            filtered_sentence = [word for word in sentence if word.lower() not in stop_words]
            filtered_sentences.append(filtered_sentence)

        st.write('Kalimat yang telah di Proses')
        # Cetak kalimat-kalimat yang telah diproses
        for filtered_sentence in filtered_sentences:
            st.write(filtered_sentence)

        # Tutup respons setelah digunakan
        response.close()

        # Menghitung jumlah kata yang diambil
        total_words = sum(len(sentence) for sentence in filtered_sentences)
        st.write('Banyak kata yang diambil')
        # Cetak jumlah kata yang diambil
        st.write(f"Jumlah kata yang diambil dari berita: {total_words}")

    with tab4:
        st.title('TF-IDF')
        # Inisialisasi penghitung TF-IDF
        tfidf_vectorizer = TfidfVectorizer()

        # Hitung TF-IDF
        tfidf_matrix = tfidf_vectorizer.fit_transform(sentences)

        # Daftar kata kunci
        feature_names = tfidf_vectorizer.get_feature_names_out()

        # Konversi matriks TF-IDF menjadi bentuk yang lebih mudah dibaca
        tfidf_values = tfidf_matrix.toarray()

        # Cetak TF-IDF untuk setiap kata dalam setiap kalimat
        for i, sentence in enumerate(sentences):
            st.write(f"Kalimat {i + 1}: {sentence}")
            for j, word in enumerate(feature_names):
                tfidf_value = tfidf_values[i][j]
                if tfidf_value > 0:
                    st.write(f"{word}: {tfidf_value:.4f}")
            st.write()

    with tab5:
        st.title('Indexing')
        # Indeks kalimat yang akan dibandingkan
        sentence1_index = 0
        sentence2_index = 1

        # Ambil vektor TF-IDF untuk kedua kalimat
        tfidf_vector1 = tfidf_matrix[sentence1_index]
        tfidf_vector2 = tfidf_matrix[sentence2_index]

        # Hitung cosine similarity antara kedua vektor
        similarity = cosine_similarity(tfidf_vector1, tfidf_vector2)

        # Cetak hasil cosine similarity
        st.write(f"Cosine Similarity antara Kalimat {sentence1_index + 1} dan Kalimat {sentence2_index + 1}: {similarity[0][0]:.4f}")

    with tab6:
        st.title('Cosine Similarity antar Kalimat')
        # Matriks TF-IDF telah dihitung sebelumnya (tfidf_matrix)
        # Hitung cosine similarity antara semua pasangan kalimat
        similarity_matrix = cosine_similarity(tfidf_matrix, tfidf_matrix)

        # Cetak hasil similarity_matrix
        num_sentences = len(sentences)  # Jumlah kalimat
        for i in range(num_sentences):
            for j in range(i+1, num_sentences):
                similarity = similarity_matrix[i][j]
                st.write(f"Cosine Similarity antara Kalimat {i + 1} dan Kalimat {j + 1}: {similarity:.4f}")

    with tab7:
        st.title('Data frame cosine Similarity')
        # Matriks TF-IDF telah dihitung sebelumnya (tfidf_matrix)
        # Hitung cosine similarity antara semua pasangan kalimat
        similarity_matrix = cosine_similarity(tfidf_matrix, tfidf_matrix)

        # Nama kolom dan indeks untuk DataFrame
        sentence_indices = [f"Kalimat {i + 1}" for i in range(len(sentences))]

        # Buat DataFrame dari hasil cosine similarity
        df = pd.DataFrame(similarity_matrix, columns=sentence_indices, index=sentence_indices)

        # Cetak DataFrame
        st.write(df)

    with tab8:
        st.title('Grafik Matrik')
        # Membuat grafik matriks
        fig, ax = plt.subplots()
        cax = ax.matshow(df, cmap='coolwarm')
        fig.colorbar(cax)

        # Memberi label pada sumbu X dan Y
        ax.set_xticks(np.arange(len(df.columns)))
        ax.set_yticks(np.arange(len(df.index)))
        ax.set_xticklabels(df.columns, rotation=90)
        ax.set_yticklabels(df.index)

        # Menampilkan nilai similarity pada matriks
        for i in range(len(df.index)):
            for j in range(len(df.columns)):
                text = ax.text(j, i, f'{df.iat[i, j]:.2f}', ha='center', va='center', color='w')

        st.pyplot()

    with tab9:
        st.title('Grafik Matriks Similarity')
        # Buat grafik dari matriks similarity
        G = nx.Graph()

        # Tambahkan simpul (node) ke grafik yang mewakili setiap kalimat
        for sentence in sentences:
            G.add_node(sentence)

        # Tambahkan tepi (edge) antara kalimat berdasarkan similarity
        for i in range(len(sentences)):
            for j in range(i + 1, len(sentences)):
                similarity = df.iloc[i, j]  # Mengambil similarity dari DataFrame
                if similarity > 0:
                    G.add_edge(sentences[i], sentences[j], weight=similarity)

        # Hitung closeness centrality untuk setiap simpul
        closeness_centrality = nx.closeness_centrality(G, distance='weight')

        # Cetak closeness centrality
        for sentence, centrality in closeness_centrality.items():
            st.write(f"Closeness Centrality of {sentence}: {centrality:.4f}")


    with tab10:
        st.title('Grafik Closeness Central')
        # Matriks TF-IDF telah dihitung sebelumnya (tfidf_matrix)
        # Hitung cosine similarity antara semua pasangan kalimat
        similarity_matrix = cosine_similarity(tfidf_matrix, tfidf_matrix)

        # Buat grafik berarah (DiGraph) berdasarkan similarity_matrix
        G = nx.DiGraph()
        for i in range(len(similarity_matrix)):
            G.add_node(i)  # Tambahkan node dengan indeks numerik

        for i in range(len(similarity_matrix)):
            for j in range(len(similarity_matrix)):
                similarity = similarity_matrix[i][j]
                if similarity > 0.1 and i != j:  # Pastikan node tidak menghubungkan dirinya sendiri
                    G.add_edge(i, j)

        # Hitung closeness centrality
        closeness_centrality = nx.closeness_centrality(G)

        # Visualisasi closeness centrality
        pos = nx.spring_layout(G)  # Atur layout grafik
        node_size = [v * 1000 for v in closeness_centrality.values()]  # Ubah ukuran node berdasarkan closeness centrality, dengan faktor pengurangan ukuran

        nx.draw_networkx_nodes(G, pos, node_size=node_size, node_color='b')
        nx.draw_networkx_edges(G, pos, edge_color='gray', arrows=True)
        nx.draw_networkx_labels(G, pos)

        st.pyplot()
        st.write("Closeness Centrality:")
        for node, closeness in closeness_centrality.items():
            st.write(f"Node {node}: {closeness:.4f}")

    with tab11:
        st.title('Page Rank Closeness Central')
        # Fungsi untuk mendapatkan indeks kalimat dengan closeness centrality terbesar
        def get_top_sentences(closeness_centrality):
            sorted_indices = np.argsort(list(closeness_centrality.values()))[::-1]
            return sorted_indices

        # Hitung cosine similarity antara semua pasangan kalimat
        similarity_matrix = cosine_similarity(tfidf_matrix, tfidf_matrix)

        # Buat grafik berarah (DiGraph) berdasarkan similarity_matrix
        G = nx.DiGraph()
        for i in range(len(similarity_matrix)):
            G.add_node(i)  # Tambahkan node dengan indeks numerik

        for i in range(len(similarity_matrix)):
            for j in range(len(similarity_matrix)):
                similarity = similarity_matrix[i][j]
                if similarity > 0.1 and i != j:
                    G.add_edge(i, j)

        # Hitung closeness centrality
        closeness_centrality = nx.closeness_centrality(G)

        # Mendapatkan indeks kalimat dengan closeness centrality terbesar
        top_sentences_indices = get_top_sentences(closeness_centrality)

        # Tampilkan nilai closeness centrality
        num_top_sentences = 3
        for i in range(num_top_sentences):
            top_sentence_index = top_sentences_indices[i]
            closeness_value = closeness_centrality[top_sentence_index]
            top_sentence = sentences[top_sentence_index]
            st.write(f"Ranking {i + 1}: Kalimat {top_sentence_index + 1} - {top_sentence}")
            st.write(f"   Closeness Centrality: {closeness_value:.4f}\n")


    with tab12:
        st.title('Page Rank antar Kalimat')
        def get_top_sentences(metric_values):
            sorted_indices = np.argsort(metric_values)[::-1]
            return sorted_indices

        # Hitung cosine similarity antara semua pasangan kalimat
        similarity_matrix = cosine_similarity(tfidf_matrix, tfidf_matrix)

        # Buat grafik berarah (DiGraph) berdasarkan similarity_matrix
        G = nx.DiGraph()
        for i in range(len(similarity_matrix)):
            G.add_node(i)  # Tambahkan node dengan indeks numerik

        for i in range(len(similarity_matrix)):
            for j in range(len(similarity_matrix)):
                similarity = similarity_matrix[i][j]
                if similarity > 0.1 and i != j:
                    G.add_edge(i, j)

        # Hitung PageRank
        pagerank = nx.pagerank(G)

        # Mendapatkan indeks kalimat dengan nilai tertinggi dari PageRank
        top_sentences_pagerank = get_top_sentences(list(pagerank.values()))

        # Tampilkan nilai PageRank
        num_top_sentences = 3
        for i in range(num_top_sentences):
            top_sentence_index = top_sentences_pagerank[i]
            pagerank_value = pagerank[top_sentence_index]
            top_sentence = sentences[top_sentence_index]
            st.write(f"Ranking {i + 1}: Kalimat {top_sentence_index + 1} - {top_sentence}")
            st.write(f"   PageRank Value: {pagerank_value:.4f}\n")

    with tab13:
        st.title('Page Rank Eigen Vect')
        # Fungsi untuk mendapatkan indeks kalimat dengan nilai tertinggi dari suatu metrik
        def get_top_sentences(metric_values):
            sorted_indices = np.argsort(metric_values)[::-1]
            return sorted_indices

        # Hitung cosine similarity antara semua pasangan kalimat
        similarity_matrix = cosine_similarity(tfidf_matrix, tfidf_matrix)

        # Buat grafik berarah (DiGraph) berdasarkan similarity_matrix
        G = nx.DiGraph()
        for i in range(len(similarity_matrix)):
            G.add_node(i)  # Tambahkan node dengan indeks numerik

        for i in range(len(similarity_matrix)):
            for j in range(len(similarity_matrix)):
                similarity = similarity_matrix[i][j]
                if similarity > 0.1 and i != j:
                    G.add_edge(i, j)

        # Hitung Eigenvector Centrality
        eigenvector_centrality = nx.eigenvector_centrality_numpy(G)

        # Mendapatkan indeks kalimat dengan nilai tertinggi dari Eigenvector Centrality
        top_sentences_eigenvector = get_top_sentences(list(eigenvector_centrality.values()))

        # Tampilkan kalimat dengan Eigenvector Centrality terbesar
        num_top_sentences = 3
        # Tampilkan nilai Eigenvector Centrality
        for i in range(num_top_sentences):
            top_sentence_index = top_sentences_eigenvector[i]
            eigenvector_value = eigenvector_centrality[top_sentence_index]
            top_sentence = sentences[top_sentence_index]
            st.write(f"Ranking {i + 1}: Kalimat {top_sentence_index + 1} - {top_sentence}")
            st.write(f"   Eigenvector Centrality: {eigenvector_value:.4f}\n")
