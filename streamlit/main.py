import streamlit as st
import tugas_1_pta
import tugas_2_topic_modelling
import tugas_3_graph
import tugas_4_klasifikai

def main():
    st.title("Aplikasi Streamlit")

    menu = ["PTA", "Topik Modeling", "Graph", "Klasifikasi"]
    choice = st.sidebar.selectbox("Pilih Program", menu)

    if choice == "PTA":
        tugas_1_pta.run()  # Panggil fungsi run() dari program1.py
    elif choice == "Topik Modeling":
        tugas_2_topic_modelling.run()  # Panggil fungsi run() dari program2.py
    elif choice == "Graph":
        tugas_3_graph.run()  # Panggil fungsi run() dari program3.py
    elif choice == "Klasifikasi":
        tugas_4_klasifikai.run()  # Panggil fungsi run() dari program4.py

if __name__ == "__main__":
    main()
