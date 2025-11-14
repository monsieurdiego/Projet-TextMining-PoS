import os
import matplotlib.pyplot as plt
import nltk
from nltk.corpus import stopwords

# Téléchargement des stop words
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# Chemin vers le dossier du Brown Corpus
dossier_brown = r"C:\Users\monsi\Desktop\Text Mining\brown"

# Structures de données
corpus_regroupe = {}
repartition_pos = {}

# Nettoyage du token
def nettoyer_token(token):
    if token.count('/') >= 1:
        parts = token.split('/')
        mot = '/'.join(parts[:-1])
        pos = parts[-1]
        pos = pos.replace('-TL', '').replace('-HL', '')
        pos = pos.split('+')[0].split('-')[0]
        return mot, pos.lower()
    return None, None

# Regroupement optimisé des PoS (10 catégories)
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

# Parcours des fichiers
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
                mot = mot.lower()

                # Suppression des stop words
                if mot in stop_words:
                    continue

                groupe = attribuer_pos_regroupe(pos)

                # Mise à jour corpus_regroupe
                if mot not in corpus_regroupe:
                    corpus_regroupe[mot] = {}
                if groupe not in corpus_regroupe[mot]:
                    corpus_regroupe[mot][groupe] = 0
                corpus_regroupe[mot][groupe] += 1

                # Mise à jour repartition_pos
                if groupe not in repartition_pos:
                    repartition_pos[groupe] = 0
                repartition_pos[groupe] += 1

# Statistiques
nombre_total_de_mots = sum(repartition_pos.values())
nombre_mots_distincts = len(corpus_regroupe)
nombre_pos_groupes = len(repartition_pos)

# Affichage
print(f"\nCorpus Brown analysé avec suppression des stop words.")
print(f"Nombre total de mots : {nombre_total_de_mots}")
print(f"Nombre de mots distincts : {nombre_mots_distincts}")
print(f"Nombre de catégories PoS regroupées : {nombre_pos_groupes}")
print(f"Catégories : {sorted(repartition_pos.keys())}")
print(f"Exemples :")
for mot in list(corpus_regroupe.keys())[:10]:
    print(f"  {mot} → {corpus_regroupe[mot]}")

# Diagramme en barres
plt.figure(figsize=(10, 6))
plt.bar(sorted(repartition_pos.keys()), [repartition_pos[pos] for pos in sorted(repartition_pos.keys())], color='mediumslateblue')
plt.title("Répartition des catégories PoS après suppression des stop words")
plt.xlabel("Catégories PoS")
plt.ylabel("Nombre d'occurrences")
plt.xticks(rotation=45)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()
