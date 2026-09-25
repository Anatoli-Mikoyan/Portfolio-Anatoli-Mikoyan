# Outil d'envoi de candidatures

Envoie à chaque entreprise d'une liste un mail contenant une lettre de motivation
personnalisée (nom de l'entreprise, et nom du contact si connu) avec le CV en pièce jointe.

## 1. Préparer les fichiers

- **`entreprises.csv`** : copie `entreprises.exemple.csv`. Colonnes : `entreprise`, `email`,
  `contact` (facultatif, ex. « Madame Dupont ») et `accroche` (une phrase propre à l'entreprise).
  Séparateur `,` ou `;` (export Excel accepté). Une colonne `envoyer` à `non` met une entreprise de côté.
- **La lettre**, écrite directement dans le corps du mail (pas de PDF), est choisie ainsi :
  1. `lettres/<nom-entreprise>.txt` s'il existe (lettre entièrement écrite pour cette entreprise) ;
  2. sinon `lettre.txt`, où `{accroche}` reçoit la phrase de la colonne `accroche` ;
  3. sinon l'entreprise est **ignorée** : aucune lettre générique ne part.

  `{entreprise}` est remplacé par le nom, `{salutation}` par « Bonjour Madame X, » ou « Bonjour, ».
  La 1re ligne `Objet : ...` devient l'objet du mail.
- **CV** : `CV_Anatoli_Mikoyan_alternance.pdf` s'il est dans ce dossier, sinon `../Cv.pdf` (ou `--cv chemin/vers/cv.pdf`).
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
python3 envoyer_candidatures.py --envoyer --a-partir "2026-09-28 08:30"   # envoi programmé
```

Avec `--a-partir`, le script attend l'heure choisie : l'ordinateur doit rester allumé et connecté.

- 45 s d'attente entre chaque mail (`--delai`) pour ne pas être classé en spam.
- Chaque envoi est noté dans `envois.log.csv` : relancer le script ne renvoie jamais
  deux fois à la même adresse.
- Conseil : pas plus de 20 à 50 envois par jour depuis un compte Gmail perso.
