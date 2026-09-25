# Outil d'envoi de candidatures

Envoie à chaque entreprise d'une liste un mail contenant une lettre de motivation
personnalisée (nom de l'entreprise, et nom du contact si connu) avec le CV en pièce jointe.

## 1. Préparer les fichiers

- **`entreprises.csv`** : copie `entreprises.exemple.csv`. Colonnes : `entreprise`, `email`,
  et `contact` (facultatif, ex. « Madame Dupont »). Séparateur `,` ou `;` (export Excel accepté).
- **`lettre.txt`** : ta lettre. `{entreprise}` est remplacé par le nom de l'entreprise,
  `{salutation}` par le contact ou « Madame, Monsieur ». La 1re ligne `Objet : ...` devient l'objet du mail.
- **CV** : par défaut `../Cv.pdf` (sinon `--cv chemin/vers/cv.pdf`).
- **`.env`** : copie `.env.exemple` et mets ton adresse + mot de passe.
  Pour Gmail, active la validation en 2 étapes puis crée un
  [mot de passe d'application](https://myaccount.google.com/apppasswords).
  Autre messagerie : adapte `SMTP_SERVEUR` (connexion SSL, port 465).

`.env`, `entreprises.csv` et le journal sont ignorés par git : ils ne seront jamais publiés.

## 2. Lancer

```bash
python3 envoyer_candidatures.py                        # aperçu de tous les mails, rien n'est envoyé
python3 envoyer_candidatures.py --test ton@mail.fr     # 1 mail d'essai envoyé à toi-même
python3 envoyer_candidatures.py --envoyer --max 20     # envoi réel, 20 entreprises max
```

- 45 s d'attente entre chaque mail (`--delai`) pour ne pas être classé en spam.
- Chaque envoi est noté dans `envois.log.csv` : relancer le script ne renvoie jamais
  deux fois à la même adresse.
- Conseil : pas plus de 20 à 50 envois par jour depuis un compte Gmail perso.
