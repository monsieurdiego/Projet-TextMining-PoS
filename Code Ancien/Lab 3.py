import os
import nltk
import pandas as pd
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from sklearn.metrics import mutual_info_score

#Ce code prend entre 20 et 60min... Complexité en ncarré  avec n =49616...
# Téléchargement des stop words
"""""""
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# Chemin vers le dossier du Brown Corpus
dossier_brown = r"C:\Users\monsi\Desktop\Text Mining\brown"

# Structures de données
corpus_regroupe = {}
repartition_pos = {}

# Nettoyage du token
def nettoyer_token(token):
    if '/' in token:
        parts = token.split('/')
        mot = '/'.join(parts[:-1])
        pos = parts[-1]
        pos = pos.replace('-TL', '').replace('-HL', '')
        pos = pos.split('+')[0].split('-')[0]
        return mot.lower(), pos.lower()
    return None, None

# Regroupement des PoS en 10 catégories
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

# Lecture et traitement du corpus
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

                if mot not in corpus_regroupe:
                    corpus_regroupe[mot] = {}
                if groupe not in corpus_regroupe[mot]:
                    corpus_regroupe[mot][groupe] = 0
                corpus_regroupe[mot][groupe] += 1

                if groupe not in repartition_pos:
                    repartition_pos[groupe] = 0
                repartition_pos[groupe] += 1

# Statistiques générales
print(f"\nCorpus Brown analysé avec suppression des stop words.")
print(f"Nombre total de mots : {sum(repartition_pos.values())}")
print(f"Nombre de mots distincts : {len(corpus_regroupe)}")
print(f"Nombre de catégories PoS regroupées : {len(repartition_pos)}")
print(f"Catégories : {sorted(repartition_pos.keys())}")
print("Exemples :")
for mot in list(corpus_regroupe.keys())[:10]:
    print(f"  {mot} → {corpus_regroupe[mot]}")

# Construction du DataFrame
rows = []
for mot, pos_dict in corpus_regroupe.items():
    total = sum(pos_dict.values())
    pos_majoritaire = max(pos_dict, key=pos_dict.get)
    rows.append({'mot': mot, 'freq': total, 'pos': pos_majoritaire})

df = pd.DataFrame(rows)

# Calcul de l'information mutuelle pour chaque mot
mi_scores = []
classes = df['pos'].tolist()
for i in range(len(df)):
    presence = [1 if j == i else 0 for j in range(len(df))]
    score = mutual_info_score(presence, classes)
    mi_scores.append(score)

df['MI_score'] = mi_scores
df_sorted = df.sort_values(by='MI_score', ascending=False)

# Affichage des scores
print("\nScores d'information mutuelle par mot :")
print(df_sorted[['mot', 'freq', 'pos', 'MI_score']].head(20))

# Filtrage des mots informatifs
seuil = 0.1
mots_pertinents = df_sorted[df_sorted['MI_score'] >= seuil]['mot'].tolist()
print(f"\nMots retenus (MI ≥ {seuil}) : {len(mots_pertinents)} mots")
print(mots_pertinents[:20])

# Visualisation de la répartition PoS
plt.figure(figsize=(10, 6))
plt.bar(sorted(repartition_pos.keys()), [repartition_pos[pos] for pos in sorted(repartition_pos.keys())], color='mediumslateblue')
plt.title("Répartition des catégories PoS après suppression des stop words")
plt.xlabel("Catégories PoS")
plt.ylabel("Nombre d'occurrences")
plt.xticks(rotation=45)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()
