"Même sur une machine performante, cette approche séquentielle devient coûteuse en temps et en mémoire. "
"Pour optimiser cela, on peut reformuler le problème en construisant une matrice où chaque mot est une colonne (feature) "
"et chaque ligne représente une instance (par exemple un document ou un mot avec sa classe). "
"En utilisant la fonction mutual_info_classif de scikit-learn sur cette matrice vectorisée, "
"on peut calculer tous les scores d'un seul coup, de manière parallèle et optimisée. "
"Cette approche réduit drastiquement le temps d'exécution, passant de plusieurs dizaines de minutes à quelques secondes, "
"sans altérer la logique ni l'objectif du traitement."

"""Pour optimiser le temps d'exécution, nous avons modifié la structure des données en regroupant 
tous les mots par catégorie grammaticale (PoS). Chaque classe devient un “document” contenant tous 
les mots qui lui sont associés. Cette approche permet de réduire drastiquement la taille de la matrice 
d’entrée, en passant de plusieurs centaines de milliers de lignes (un mot par ligne) à seulement dix lignes 
(une par PoS). En vectorisant ces documents avec CountVectorizer et en appliquant mutual_info_classif, 
nous obtenons les scores d'information mutuelle de manière vectorisée, rapide et efficace. Cette optimisation 
permet de passer d’un temps d’exécution de plusieurs minutes à quelques secondes, 
sans altérer la logique ni l’objectif du traitement."""

import os
import nltk
import pandas as pd
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_selection import mutual_info_classif
from sklearn.preprocessing import LabelEncoder

# Téléchargement des stop words
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# Chemin vers le dossier du Brown Corpus
dossier_brown = r"C:\Users\monsi\Desktop\Text Mining\brown"

# Structures de regroupement
repartition_pos = {}

def nettoyer_token(token):
    if '/' in token:
        parts = token.split('/')
        mot = '/'.join(parts[:-1])
        pos = parts[-1]
        pos = pos.replace('-TL', '').replace('-HL', '')
        pos = pos.split('+')[0].split('-')[0]
        return mot.lower(), pos.lower()
    return None, None

def attribuer_pos_regroupe(pos):
    if pos in {'nn', 'nns', 'np', 'np$', 'nps', 'nps$', 'nr', 'nrs', 'nn$', 'nnp', 'nnpc', 'nna', 'nnc'}:
        return 'NOUN'
    elif pos in {'vb', 'vbd', 'vbg', 'vbn', 'vbz', 'vba', 'bez', 'bed', 'bedz', 'beg', 'bem', 'ben', 'ber',
                 'do', 'dod', 'doz', 'hv', 'hvd', 'hvg', 'hvn', 'hvz', 'md'}:
        return 'VERB'
    elif pos.startswith('jj'):
        return 'ADJ'
    elif pos in {'rb', 'rbr', 'rbs', 'rbt', 'rn', 'rp', 'wrb'}:
        return 'ADV'
    elif pos in {'pp$', 'pp$$', 'ppl', 'ppls', 'ppo', 'pps', 'ppss', 'pn', 'pn$', 'wp$', 'wpo', 'wps',
                 'prp', 'prps', 'prp$'}:
        return 'PRON'
    elif pos in {'at', 'dt', 'dts', 'dti', 'dtx', 'abl', 'abn', 'abx', 'ap'}:
        return 'DET'
    elif pos in {'in', 'fw-in'}:
        return 'PREP'
    elif pos in {'cc', 'cs', 'dtx'}:
        return 'CONJ'
    elif pos in {'cd', 'od'}:
        return 'NUM'
    else:
        return 'OTHER'

# Lecture et regroupement des mots par PoS
for nom_fichier in os.listdir(dossier_brown):
    chemin_fichier = os.path.join(dossier_brown, nom_fichier)
    if os.path.isfile(chemin_fichier):
        try:
            with open(chemin_fichier, "r", encoding="utf-8") as fichier:
                contenu = fichier.read()
        except UnicodeDecodeError:
            with open(chemin_fichier, "r", encoding="latin-1") as fichier:
                contenu = fichier.read()

        tokens = contenu.split()

        for token in tokens:
            mot, pos = nettoyer_token(token)
            if mot and pos and pos != 'nil':
                if mot in stop_words:
                    continue
                groupe = attribuer_pos_regroupe(pos)
                if groupe not in repartition_pos:
                    repartition_pos[groupe] = []
                repartition_pos[groupe].append(mot)

# Construction des documents par PoS
documents = [' '.join(repartition_pos[pos]) for pos in repartition_pos]
classes = list(repartition_pos.keys())

print(f"\nCorpus regroupé par PoS. Nombre de classes : {len(classes)}")
print(f"Catégories : {classes}")

# Vectorisation
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(documents)
feature_names = vectorizer.get_feature_names_out()

# Encodage des classes
le = LabelEncoder()
y = le.fit_transform(classes)

# Calcul de l'information mutuelle
mi_scores = mutual_info_classif(X, y, discrete_features=True) #One word associat

# Résultats
df_mi = pd.DataFrame({
    'mot': feature_names,
    'MI_score': mi_scores
}).sort_values(by='MI_score', ascending=False)

print("\nScores d'information mutuelle par mot :")
print(df_mi.head(20))

# Filtrage
seuil = 0.1
mots_pertinents = df_mi[df_mi['MI_score'] >= seuil]['mot'].tolist()
print(f"\nMots retenus (MI ≥ {seuil}) : {len(mots_pertinents)} mots")
print(mots_pertinents[:20])

# Visualisation
df_classes = pd.Series([len(repartition_pos[pos]) for pos in classes], index=classes)
plt.figure(figsize=(10, 6))
plt.bar(df_classes.index, df_classes.values, color='mediumslateblue')
plt.title("Répartition des mots par catégorie PoS")
plt.xlabel("Catégories PoS")
plt.ylabel("Nombre de mots")
plt.xticks(rotation=45)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()
