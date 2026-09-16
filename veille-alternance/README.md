# Veille alternance

Un outil qui va chercher tous les matins les offres d'alternance récemment
publiées, les classe selon mon CV et mon projet de formation, et m'envoie les
meilleures par e-mail.

**Profil ciblé** — Développeur IA / Python / Data, contrat d'apprentissage ou de
professionnalisation de 12 mois, rentrée octobre 2026 (Bachelor IA 3ᵉ année,
ETNA Ivry-sur-Seine). France entière : l'école n'impose qu'un vendredi toutes
les trois semaines sur site, donc la province et le télétravail restent
parfaitement jouables.

Tableau de bord : <https://anatoli-mikoyan.github.io/Portfolio-Anatoli-Mikoyan/veille-alternance/>

---

## Ce que fait l'outil

1. **Collecte** les annonces sur plusieurs sources (France Travail, La Bonne
   Alternance, Adzuna) à partir de 10 requêtes métier.
2. **Filtre** sur la fraîcheur (7 jours par défaut) et vérifie lui-même qu'il
   s'agit bien d'une alternance — les filtres des API se trompent.
3. **Dédoublonne**, y compris entre sources : la même annonce republiée avec une
   ponctuation différente ou une mention `(H/F)` n'apparaît qu'une fois.
4. **Classe** chaque offre selon le CV : poste visé, technologies maîtrisées,
   fraîcheur, localisation. Les organismes de formation qui publient des
   annonces pour vendre leur cursus et les offres exigeant un Bac+5 sont
   dégradés.
5. **Mémorise** ce qui a déjà été envoyé — un digest ne répète jamais une offre.
6. **Restitue** : e-mail, tableau de bord HTML, `digest.md`, `etat/dernier-digest.json`.

## Essayer tout de suite, sans rien configurer

```bash
cd veille-alternance
pip install -r requirements.txt
python veille.py --demo
```

Un jeu d'essai local traverse toute la chaîne et génère `index.html`. Aucun
appel réseau.

```bash
python test_veille.py   # 13 tests hors ligne sur le tri et la déduplication
python veille.py --sonde  # que suis-je capable de joindre, et avec quelles clés ?
```

## Mise en service (environ 15 minutes)

Les trois sources sont indépendantes : **une seule suffit pour démarrer**, et
France Travail est de loin la plus fournie en alternance. L'outil ignore
proprement les sources non configurées.

### 1. France Travail — prioritaire, gratuit

1. Créer un compte sur <https://francetravail.io/>
2. Créer une application, puis s'abonner à l'API **« Offres d'emploi v2 »**
3. Relever le `client_id` et le `client_secret`

### 2. La Bonne Alternance — gratuit, spécialisé alternance

Demander une clé sur <https://api.apprentissage.beta.gouv.fr/> (API du service
public dédiée à l'alternance, elle voit aussi des entreprises qui recrutent sans
avoir publié d'annonce).

### 3. Adzuna — gratuit, agrégateur

Créer un compte sur <https://developer.adzuna.com/> pour obtenir `app_id` et
`app_key`. Utile pour ratisser les job boards privés.

### 4. E-mail (Gmail)

La validation en deux étapes doit être active, puis créer un mot de passe
d'application sur <https://myaccount.google.com/apppasswords>. C'est ce
mot de passe à 16 caractères qui sert de `SMTP_PASS`, **jamais** le mot de passe
du compte.

### 5. Déclarer les secrets sur GitHub

`Settings` → `Secrets and variables` → `Actions` → `New repository secret` :

| Secret | Utilité |
| --- | --- |
| `FT_CLIENT_ID` / `FT_CLIENT_SECRET` | France Travail |
| `LBA_API_KEY` | La Bonne Alternance |
| `ADZUNA_APP_ID` / `ADZUNA_APP_KEY` | Adzuna |
| `SMTP_USER` | Adresse Gmail d'envoi |
| `SMTP_PASS` | Mot de passe d'application Gmail |
| `SMTP_HOST` | Facultatif, `smtp.gmail.com` par défaut |

### 6. Vérifier

Onglet `Actions` → `Veille alternance` → `Run workflow` → mode **`sonde`**.
Le diagnostic affiche les clés reconnues et les endpoints joignables, sans rien
envoyer. Ensuite, mode `normal` pour un vrai tour de veille.

Une fois les secrets en place, le workflow tourne **tous les jours à 7h00** et
pousse le tableau de bord à jour dans le dépôt.

## Ajuster la veille

Tout se règle dans **`profil.json`**, seul fichier à éditer :

| Réglage | Effet |
| --- | --- |
| `metiers_vises` | Intitulés qui déclenchent le plus gros bonus |
| `mots_cles_recherche` | Requêtes envoyées aux API |
| `competences` | Technologies valorisées, par famille et par poids |
| `exclusions` | Ce qui est dégradé ou rejeté |
| `geographie` | `france_entiere`, rayon, départements prioritaires |
| `seuil_score_minimum` | Plus haut = moins d'offres, mieux ciblées |
| `recherche.nb_offres_digest` | Taille du digest (20 par défaut) |

Un terme suffixé par `*` est un préfixe volontaire : `comptab*` attrape
« comptable » et « comptabilité ». Sans l'étoile, la comparaison se fait sur des
mots entiers — c'est ce qui évite que `ia` se déclenche sur « commerc**ia**l ».

**Trop d'offres hors sujet ?** Monter `seuil_score_minimum`, ou ajouter le terme
gênant dans `exclusions.malus_metier`.
**Pas assez d'offres ?** Baisser le seuil, élargir `jours_recence_max`, ajouter
des `mots_cles_recherche`.

## Commandes

```bash
python veille.py                      # collecte + tableau de bord (sans e-mail)
python veille.py --email              # + envoi du digest
python veille.py --jours 3 --max 15   # 15 offres publiées dans les 3 derniers jours
python veille.py --sources france-travail
python veille.py --ignorer-historique # réafficher des offres déjà envoyées
python veille.py --demo               # jeu d'essai hors ligne
python veille.py --sonde              # diagnostic clés + réseau
python veille.py -v                   # journal détaillé
```

## Structure

```
veille-alternance/
├── profil.json      # le profil issu du CV — le fichier à éditer
├── veille.py        # orchestration et interface en ligne de commande
├── sources.py       # connecteurs France Travail / LBA / Adzuna
├── scoring.py       # classement d'une offre selon le profil
├── historique.py    # mémoire : ne jamais renvoyer deux fois la même offre
├── modele.py        # objet Offre commun à toutes les sources
├── rendu.py         # Markdown, tableau de bord HTML, e-mail
├── test_veille.py   # tests hors ligne
├── fixtures/        # jeu d'essai
└── etat/            # historique et dernier digest (écrits par l'outil)
```

## Limites connues

- **Le schéma des API bouge.** Les connecteurs lisent chaque champ à plusieurs
  emplacements possibles et n'échouent jamais bruyamment : si une source change,
  les autres continuent. En cas de récolte vide, lancer `--sonde` en premier.
- **Les codes de contrat de France Travail** (`E2` apprentissage, `FS`
  professionnalisation) sont surchargeables via `FT_NATURES_CONTRAT`. Un
  contrôle maison vérifie de toute façon que chaque offre est bien une
  alternance.
- **Le score n'a pas de valeur absolue**, il ne sert qu'à trier. Un 128 n'est pas
  « deux fois mieux » qu'un 64.
- **Indeed et LinkedIn n'ont pas d'API publique** exploitable ici. Les offres qui
  n'existent que là ne remonteront pas.
