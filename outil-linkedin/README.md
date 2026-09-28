# Prospection LinkedIn (semi-automatique)

Rien n'est envoyé automatiquement : LinkedIn interdit les robots et peut suspendre le compte.
Ces outils préparent tout, toi tu cliques. Compte 10 secondes par personne.

## 1. Trouver les prospects : `recherche_prospects.html`
Ouvre la page (double-clic), connecte-toi à LinkedIn dans le même navigateur, puis :
- **Recherches générales** : posts récents d'entreprises qui cherchent un alternant, offres LinkedIn, recruteurs tech.
- **Tes entreprises** : pour chaque entreprise déjà contactée par mail, un lien « Recruteurs / RH » et un lien « Tech ».
Pour chaque bonne personne, copie l'adresse de son profil dans `prospects.csv`.

Pour régénérer la page avec de nouvelles listes :
```
python chercher_prospects.py ..\outil-candidatures\entreprises-lot4.csv
```

## 2. Préparer les invitations : `prospects.csv` puis `linkedin.html`
Copie `prospects.exemple.csv` en `prospects.csv` et remplis une ligne par personne :
`prenom;nom;entreprise;poste;type;profil;accroche`
- `type` : `rh` (recruteur) ou `tech` (CTO, lead dev, manager).
- `profil` : l'adresse du profil LinkedIn.
- `accroche` (facultatif) : une phrase sur la personne, par exemple « j'ai vu votre post sur les agents IA ».

Puis :
```
python preparer_linkedin.py
```
La page `linkedin.html` s'ouvre : une fiche par personne avec « Ouvrir le profil », la note prête à copier
(200 caractères maximum, la limite d'un compte gratuit), une case « Envoyé » qui reste cochée, et le message à
envoyer quand l'invitation est acceptée.

## Règles pour ne pas être bloqué
- 15 à 20 invitations par jour maximum, jamais plus de 100 par semaine.
- Toujours une note personnalisée, jamais d'invitation vide en masse.
- Si beaucoup d'invitations restent en attente, retire les plus anciennes (Réseau, puis Invitations envoyées).
