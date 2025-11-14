import os

# Chemin vers le fichier ca01
chemin_fichier = r"C:\Users\monsi\Desktop\Text Mining\brown\ca01"

# Dictionnaire pour stocker les occurrences {mot: {PoS: count}}
corpus_stats = {}

def nettoyer_token(token):
    """
    Extrait le mot et le PoS d'un token de type 'word/tag' ou 'word/part1/part2/tag'.
    Gère les cas avec plusieurs '/' et nettoie les suffixes comme -TL, -HL, etc.
    """
    if token.count('/') >= 1:
        parts = token.split('/')
        mot = '/'.join(parts[:-1])
        pos = parts[-1]

        # Nettoyage du tag
        pos = pos.replace('-TL', '').replace('-HL', '')
        pos = pos.split('+')[0]  # Garde la partie avant '+'
        pos = pos.split('-')[0]  # Garde la partie avant '-'
        return mot, pos
    return None, None

try:
    with open(chemin_fichier, "r", encoding="utf-8") as fichier:
        contenu = fichier.read()
        tokens = contenu.split()

        for token in tokens:
            mot, pos = nettoyer_token(token)
            if mot and pos and pos.lower() != 'nil':
                if mot not in corpus_stats:
                    corpus_stats[mot] = {}
                if pos not in corpus_stats[mot]:
                    corpus_stats[mot][pos] = 0
                corpus_stats[mot][pos] += 1

except UnicodeDecodeError:
    with open(chemin_fichier, "r", encoding="latin-1") as fichier:
        contenu = fichier.read()
        tokens = contenu.split()
        for token in tokens:
            mot, pos = nettoyer_token(token)
            if mot and pos and pos.lower() != 'nil':
                if mot not in corpus_stats:
                    corpus_stats[mot] = {}
                if pos not in corpus_stats[mot]:
                    corpus_stats[mot][pos] = 0
                corpus_stats[mot][pos] += 1

# Statistiques globales
nombre_total_de_mots = sum(sum(pos_counts.values()) for pos_counts in corpus_stats.values())
nombre_mots_distincts = len(corpus_stats)
pos_distincts = set(pos for pos_counts in corpus_stats.values() for pos in pos_counts)
nombre_pos_distincts = len(pos_distincts)

print(f" Fichier ca01 analysé.")
print(f"Nombre total de mots : {nombre_total_de_mots}")
print(f"Nombre de mots distincts : {nombre_mots_distincts}")
print(f"Nombre de PoS distincts : {nombre_pos_distincts}")
print(f"Exemples :")
for mot in list(corpus_stats.keys())[:10]:
    print(f"  {mot} → {corpus_stats[mot]}")
