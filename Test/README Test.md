# Principe des tests

Ce dossier fournit 2 exemples complets de calcul de l'indicateur d'integrite fonctionnelle de la biosphere sur le territoire de Montbrison.

Les deux scenarios couvrent les deux workflows principaux du depot :
- un test raster avec `Integrity_Raster.py`,
- un test vectoriel avec `Intersection_Vectors.py` puis `Integrity_Vector.py`.

# Deroule des tests

## Test 1 - `Integrity_Raster.py`

### Contenu du dossier pour ce test

- `Raster_layer_CLC.tif` : couche raster de couverture des sols a analyser.
- `Raster_layer_CLC.qml` : fichier de style pour representer la couche raster.
- `Emprise.gpkg` : zone sur laquelle faire le calcul d'integrite.
- `CLCplus_Backbone_2023_Documentation.pdf` : documentation des classes presentes dans la couche raster.

### Calcul de l'integrite fonctionnelle de la biosphere

Nous proposons la classification suivante pour `Raster_layer_CLC.tif` :

| Number | Land cover | Classification binaire |
| --- | --- | --- |
| 1 | Sealed | 0 |
| 2 | Woody needle leaved trees | 1 |
| 3 | Woody broadleaved deciduous trees | 1 |
| 4 | Woody broadleaved evergreen trees | 1 |
| 5 | Low-growing woody plants | 1 |
| 6 | Permanent herbaceous | 1 |
| 7 | Periodically herbaceous | 0 |
| 8 | Lichens and mosses | 1 |
| 9 | Non- and sparsely vegetated | NA |
| 10 | Water | NA |
| 11 | Snow and ice | NA |

Cela se traduit par le bloc de commande suivant a entrer dans `Integrity_Raster.py` :

```python
CLASSES_1: str = "2 3 4 5 6 8"   # semi-natural = 1
CLASSES_0: str = "1 7"           # non-semi-natural = 0
CLASSES_NULL: str = "9 10 11"    # ignored = NaN
```

### Output attendu

Les outputs attendus pour ce test sont fournis dans `Output_test_1` :
- `binary_output.tif`
- `integrity_output.tif`
- `histogram.csv`
- `histogram.png`
- `execution.log`
- `Representation_integrité.qml`

## Test 2 - `Intersection_Vectors.py` + `Integrity_Vector.py`

### Contenu du dossier pour ce test

- `Primary_vector_layer_COSIA.gpkg` : couche principale, champ analyse `numero`.
- `Secondary_vector_layer_CarHab.gpkg` : couche secondaire, champ analyse `code_physio`.
- `Primary_layer_COSIA.qml` : style de representation de la couche COSIA.
- `Secondary_layer_CarHab.qml` : style de representation de la couche CarHab.
- `Cosia_Documentation_Technique_IGN_2023.pdf` : documentation des classes COSIA.
- `CarHab_42_Loire_Notice.pdf` : documentation des classes CarHab.
- `Emprise.gpkg` : zone sur laquelle faire le calcul d'integrite.

### Intersection COSIA - CarHab - `Intersection_Vectors.py`

La premiere etape consiste a croiser les 2 bases de donnees vectorielles d'usages des sols :
- [COSIA](https://cartes.gouv.fr/rechercher-une-donnee/dataset/IGNF_COSIA)
- [CarHab](https://cartes.gouv.fr/rechercher-une-donnee/dataset/INPN-CARHAB_HABITATS)

L'objectif de ce croisement est d'ameliorer la precision de la couche de couverture des sols utilisee ensuite pour le calcul de l'indicateur.

Le principe est d'utiliser COSIA comme couche principale car elle donne une analyse fine des couvertures arborees. En revanche, la categorie `12` (`Pelouse`) melange des situations qui doivent ensuite etre classees soit en `1` (semi-naturel), soit en `0` (domine par l'Homme). Une intersection avec CarHab permet de raffiner cette categorie.

Exemple de regles :
- si `COSIA = 12` et `CarHab = 3301` (`Prairie fauchee`) ou `3302` (`Prairie paturee`) ou `8001` (`Prairie temporaire`), alors `Code_Primary_Secondary = 120` ;
- si `COSIA = 12` et `CarHab = 3200` (`Vegetation herbacee haute`) ou `3300` (`Prairie indeterminee`), alors `Code_Primary_Secondary = 121` ;
- sinon, la valeur d'origine de COSIA est conservee.

Cela se traduit par le bloc de commande suivant a entrer dans `Intersection_Vectors.py` :

```python
RULES: list[dict[str, Any]] = [
    {"primary": 12, "secondary_in": {3301, 3302, 8001}, "secondary_not_in": None, "output": 120},
    {"primary": 12, "secondary_in": {3200, 3300}, "secondary_not_in": None, "output": 121},
]
```

Le programme cree une nouvelle couche avec le champ `Code_Primary_Secondary`.

Important :
- les regles sont evaluees dans l'ordre ;
- si plusieurs regles correspondent a un meme polygone, la derniere regle applicable l'emporte ;
- si cette sortie est ensuite utilisee dans `Integrity_Vector.py`, les valeurs ecrites dans `Code_Primary_Secondary` doivent rester entieres ou convertibles en entiers.

### Calcul de l'integrite fonctionnelle de la biosphere - `Integrity_Vector.py`

Une fois la base de donnees de couverture des sols construite, le calcul de l'integrite fonctionnelle de la biosphere peut etre realise. Nous proposons la classification suivante :

| Numero de la couche d'intersection | Couverture | Classification binaire |
| --- | --- | --- |
| 1 | Batiment | 0 |
| 2 | Zone permeable | 0 |
| 3 | Zone impermeable | 0 |
| 4 | Piscine | 0 |
| 5 | Sol nu | NA |
| 6 | Surface eau | NA |
| 7 | Neige | NA |
| 8 | Conifere | 1 |
| 9 | Feuillu | 1 |
| 10 | Broussaille | 1 |
| 11 | Vigne | 0 |
| 120 | Paturage | 0 |
| 121 | Prairie naturelle | 1 |
| 12 | Pelouse urbaine | 0 |
| 13 | Culture | 0 |
| 14 | Terre labouree | 0 |
| 15 | Serre | 0 |

Cela se traduit par le bloc de commande suivant a entrer dans `Integrity_Vector.py` :

```python
CLASSES_1: str = "8 9 10 121"
CLASSES_0: str = "1 2 3 4 11 12 13 14 15 120"
CLASSES_NULL: str = "5 6 7"
```

### Output attendu

Les outputs attendus pour ce test sont fournis dans `Output_test_2` :
- `Output_Intersection_COSIA_CarHab.gpkg`
- `Output_Rasterization_COSIA_CarHab.tif`
- `binary_output.tif`
- `integrity_output.tif`
- `histogram.csv`
- `histogram.png`
- `execution.log`
- `Representation_integrité.qml`
