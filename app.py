import streamlit as st
import joblib  
import time

import streamlit as st

# Configuration de la page
st.set_page_config(page_title="Détecteur IA vs Humain", page_icon=":robot_face:", layout="wide")

@st.cache_data
def load_model():
    model = joblib.load('final_logistic_regression_model.pkl')
    return model

model = load_model()

def predict(text):
    # Cette fonction prend en entrée un texte et retourne la prédiction du modèle
    return model.predict([text])[0]


# En-tête principale
st.title('Détecteur IA :robot_face: vs Humain :smiley:')
st.markdown("""
Cette application utilise le machine learning pour prédire si un texte a été généré par une IA (GPT) ou écrit par un humain.
Entrez un texte ci-dessous et laissez l'IA analyser si le contenu semble être écrit par une intelligence artificielle ou par une personne réelle.
""")

# Entrée de texte par l'utilisateur
with st.container():
    user_input = st.text_area("Entrez le texte à analyser ici:", height=150, placeholder="Tapez ou collez du texte ici...")

# Bouton de prédiction
if st.button('Prédire :mag:'):
    with st.spinner('Analyse en cours...'):
        start_time = time.time()
        label = predict(user_input)
        end_time = time.time()
        st.write(f"Temps d'inférence: {end_time - start_time} secondes")
        
    if label == 0:
        st.success('### Résultat: Texte généré par GPT :robot_face:')
    else:
        st.success('### Résultat: Texte écrit par un humain :smiley:')

# À propos de l'application
with st.expander("ℹ️ - À propos de cette application"):
    st.write("""
    Cette application fait partie d'un projet de recherche sur la capacité des modèles de machine learning à distinguer les textes générés par des humains de ceux générés par des algorithmes de type GPT. Les modèles ont été entraînés sur des ensembles de données variés pour améliorer leur précision.
    """)

# Footer
st.markdown("---")
st.markdown("🚀 Application développée par Julien OHANA, Prisca RAMANANTOANINA, Sofiane EL FARTASS, Enzo NATALI, Maël REYNAUD")

