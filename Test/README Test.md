# Principe du test

Ce test donne un exemple complet d'un calcul de l'indicateur d'intégrité fonctionnelle de la biosphère. Sont fournies dans le dossier des bases de données de couvertures des sols au niveau de la ville de Montbrison sur lesquels tester le calcul de l'indicateur.

# Déroulé du test

## Contenu du dossier

- `Primary_layer_COSIA.gpkg` : champ analyse `numero`
- `Secondary_layer_CarHab.gpkg` : champ analysé `code_physio`
- `Primary_layer_COSIA.qml` : fichier de style pour représenter la légende fournie dans la notice de la couche `numero`
- `Secondary_layer_CarHab.qml` : fichier de style pour représenter la légende fournie dans la notice de la couche `code_physio`
- `Emprise.gpkg` : zone sur laquelle faire le calcul d'intégrité

## Intersection COSIA - CarHab - `Intersection_Vector`

Nous proposons comme première étape de réaliser une intersection entre 2 bases de données vecteur d'usages des sols : [COSIA](https://cartes.gouv.fr/rechercher-une-donnee/dataset/IGNF_COSIA) et [CarHab](https://cartes.gouv.fr/rechercher-une-donnee/dataset/INPN-CARHAB_HABITATS). L'objectif de ce croisement est d'augmenter la précision de la base de données de couverture des sols utilisée pour le calcul.

Le principe est d'utiliser COSIA comme couche principale car elle donne une analyse fine des couvertures arborées. Cependant, COSIA est imprécise sur les couches herbacées : dans la catégorie 12 (`Pelouse`), ne sont pas différenciés les prairies (classées en `1` - semi-naturelles) des zones d'agriculture intensives (classées en dominées par l'Homme - `0`). Une intersection avec CarHab est donc proposée pour améliorer l'analyse : la catégorie `12` est intersectée avec CarHab avec la règle suivante :

- Si `COSIA = 12` et `CarHab = 3301` (`Prairie fauchée`) ou `3302` (`Prairie pâturée`) ou `8001` (`Prairie temporaire`) alors `COSIA_CarHab = 120` (`zones d'agriculture intensives`)
- Si `COSIA = 12` et `CarHab = 3200` (`Vegetation herbacée haute`) ou `3300` (`Prairie indéterminée`) ou `8001` alors `COSIA_CarHab = 120` (`prairie`)
- Sinon `COSIA = 12` (`pelouse hors prairie/pâturage, généralement des jardins particuliers`)

Cela se traduit par le bloc de commande à entrer dans `Intersection_Vector.py` :

```python
RULES: list[dict[str, Any]] = [
    {"primary": 12, "secondary_in": {3301, 3302, 8001}, "secondary_not_in": None, "output": 120},
    {"primary": 12, "secondary_in": {3200, 3300}, "secondary_not_in": None, "output": 121},
]
```

Le programme créé une nouvelle couche avec le champ `Code_Primary_Secondary` contenant cette intersection.

## Calcul de l'intégrité fonctionnelle de la biosphère - `Integrity_Vector`

Ensuite une fois la base de données de couverture des sols construite, le calcul de l'intégrité fonctionnelle de la biosphère peut être réalisé. Nous proposons la classification suivante :

| Numéro de la couche d'intersection | Couverture | Classification binaire |
| --- | --- | --- |
| 1 | Batiment | 0 |
| 2 | Zone perméable | 0 |
| 3 | Zone imperméable | 0 |
| 4 | Piscine | 0 |
| 5 | Sol nu | NA |
| 6 | Surface eau | NA |
| 7 | Neige | NA |
| 8 | Conifère | 1 |
| 9 | Feuillu | 1 |
| 10 | Broussaille | 1 |
| 11 | Vigne | 0 |
| 120 | Pâturage | 0 |
| 121 | Prairie naturelle | 1 |
| 12 | Pelouse urbaine | 0 |
| 13 | Culture | 0 |
| 14 | Terre labourée | 0 |
| 15 | Serre | 0 |

Cela se traduit par le bloc de commande suivant à entrer dans `Integrity_Vector` :

```python
CLASSES_1: str = "8 9 10 121"
CLASSES_0: str = "1 2 3 4 11 12 13 14 15 120"
CLASSES_NULL: str = "5 6 7"
```

## Output

Les outputs attendus sont mis dans le dossier `Output`.
Un fichier `qml` est fournit pour offrir un exemple de style de représentation des valeurs d'intégrité.
