import matplotlib.pyplot as plt

import os

#  Étape 1 : définir le chemin vers le dossier contenant les fichiers du Brown Corpus
dossier_brown = r"C:\Users\monsi\Desktop\Text Mining\brown"

#  Étape 2 : créer une structure pour stocker les statistiques
# Format : {mot: {PoS_groupé: nombre d'occurrences}}
corpus_regroupe = {}

# Étape 3 : fonction pour nettoyer chaque token et extraire le mot + PoS
def nettoyer_token(token):
    """
    Un token est de la forme 'mot/tag' ou 'mot/part1/part2/tag'.
    Cette fonction extrait le mot et le tag PoS, en nettoyant les suffixes inutiles.
    """
    if token.count('/') >= 1:
        parts = token.split('/')
        mot = '/'.join(parts[:-1])  # tout sauf le dernier élément
        pos = parts[-1]             # dernier élément = tag PoS

        # Nettoyage du tag : on enlève les suffixes comme -TL, -HL, +, etc.
        pos = pos.replace('-TL', '').replace('-HL', '')
        pos = pos.split('+')[0]
        pos = pos.split('-')[0]
        return mot, pos
    return None, None

#  Étape 4 : fonction pour regrouper les tags PoS en 10 grandes catégories
def regrouper_pos(tag):
    """
    Regroupe les tags PoS détaillés en 10 catégories principales.
    """
    tag = tag.lower()
    if tag in {'nn', 'nns', 'np', 'np$', 'nps', 'nps$', 'nr', 'nrs', 'nn$', 'nnp', 'nnpc'}:
        return 'NOUN'
    elif tag in {'vb', 'vbd', 'vbg', 'vbn', 'vbz', 'vba', 'bez', 'bed', 'bedz', 'beg', 'bem', 'ben', 'ber', 'do', 'dod', 'doz', 'md', 'hv', 'hvd', 'hvg', 'hvn', 'hvz'}:
        return 'VERB'
    elif tag in {'jj', 'jjt', 'jjr', 'jjc', 'jjcc', 'jja', 'jjm', 'jjf'}:
        return 'ADJ'
    elif tag in {'rb', 'rbr', 'rbs', 'rbt', 'wrb'}:
        return 'ADV'
    elif tag in {'pps', 'pp$', 'pp$$', 'ppl', 'ppls', 'ppo', 'pn', 'pn$', 'wp$', 'wpo', 'wps', 'prp', 'prps', 'prp$'}:
        return 'PRON'
    elif tag in {'at', 'dt', 'dts', 'dti', 'dtx', 'abn', 'abx', 'abl'}:
        return 'DET'
    elif tag in {'in', 'fw-in'}:
        return 'PREP'
    elif tag in {'cc', 'cs'}:
        return 'CONJ'
    elif tag in {'uh'}:
        return 'INTJ'
    else:
        return 'OTHER'  #Show OTHER

#  Étape 5 : parcourir tous les fichiers du dossier
for nom_fichier in os.listdir(dossier_brown):
    chemin_fichier = os.path.join(dossier_brown, nom_fichier)

    # Vérifier que c'est bien un fichier (pas un dossier)
    if os.path.isfile(chemin_fichier):
        # Ouvrir le fichier avec encodage UTF-8, ou latin-1 en secours
        try:
            with open(chemin_fichier, "r", encoding="utf-8") as fichier:
                contenu = fichier.read()
        except UnicodeDecodeError:
            with open(chemin_fichier, "r", encoding="latin-1") as fichier:
                contenu = fichier.read()

        # Séparer le texte en tokens
        tokens = contenu.split()

        #  Étape 6 : traiter chaque token
        for token in tokens:
            mot, pos = nettoyer_token(token)
            if mot and pos and pos.lower() != 'nil':
                groupe = regrouper_pos(pos)

                # Ajouter au dictionnaire
                if mot not in corpus_regroupe:
                    corpus_regroupe[mot] = {}
                if groupe not in corpus_regroupe[mot]:
                    corpus_regroupe[mot][groupe] = 0
                corpus_regroupe[mot][groupe] += 1

#  Étape 7 : calculer les statistiques globales
nombre_total_de_mots = sum(sum(pos_counts.values()) for pos_counts in corpus_regroupe.values())
nombre_mots_distincts = len(corpus_regroupe)
pos_groupes = set(pos for pos_counts in corpus_regroupe.values() for pos in pos_counts)
nombre_pos_groupes = len(pos_groupes)

#  Étape 8 : afficher les résultats
print(f" Corpus Brown analysé avec regroupement des PoS.")
print(f"Nombre total de mots : {nombre_total_de_mots}")
print(f"Nombre de mots distincts : {nombre_mots_distincts}")
print(f"Nombre de catégories PoS regroupées : {nombre_pos_groupes}")
print(f"Catégories : {sorted(pos_groupes)}")
print(f"Exemples :")
for mot in list(corpus_regroupe.keys())[:10]:
    print(f"  {mot} → {corpus_regroupe[mot]}")

# Étape 9 : calculer la répartition totale par catégorie PoS
repartition_pos = {}
for pos_counts in corpus_regroupe.values():
    for pos, count in pos_counts.items():
        if pos not in repartition_pos:
            repartition_pos[pos] = 0
        repartition_pos[pos] += count

# Étape 10 : tracer le diagramme en barres
categories = sorted(repartition_pos.keys())
valeurs = [repartition_pos[pos] for pos in categories]

plt.figure(figsize=(10, 6))
plt.bar(categories, valeurs, color='skyblue')
plt.title("Répartition des catégories PoS dans le Brown Corpus")
plt.xlabel("Catégories PoS")
plt.ylabel("Nombre d'occurrences")
plt.xticks(rotation=45)
plt.tight_layout()
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()
