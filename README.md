# Faster R-CNN (ResNet-50-FPN) sur COCO

TP de détection d'objets — Master MQL, Faculté des Sciences de Tétouan.

Implémentation, entraînement et évaluation d'un détecteur **Faster R-CNN** avec un backbone **ResNet-50 + FPN** pré-entraîné sur ImageNet, sur le dataset **COCO 2017** (80 classes), avec PyTorch / torchvision.

## Contenu

| Fichier | Description |
|---|---|
| `TP_FasterRCNN_ResNet50_COCO_Kaggle.ipynb` | Notebook complet, commenté en français, prêt pour Kaggle (GPU) |
| `requirements.txt` | Dépendances pour une exécution locale |

## Étapes du notebook

1. Configuration de l'environnement et des hyper-paramètres
2. Préparation des données COCO : localisation du dataset, sous-ensembles reproductibles, exploration (classes, tailles, densité)
3. `Dataset` PyTorch et `DataLoader` (conversion des boîtes, flip horizontal)
4. Architecture Faster R-CNN : ResNet-50 + FPN, RPN, RoI Align, tête de classification / régression
5. Entraînement : SGD, warm-up + décroissance cosinus, précision mixte (AMP), gradient clipping
6. Courbes d'apprentissage (pertes, LR, mAP par époque)
7. Évaluation COCO : mAP@[.5:.95], AP@.5, AP@.75, AP par taille, AR, courbes précision-rappel, AP par classe, analyse des scores
8. Exemples de détection (meilleurs cas, cas difficiles, images d'entraînement)
9. Comparaison avec le modèle de référence torchvision entraîné sur COCO complet
10. Conclusion et pistes d'amélioration

Chaque bloc de code est suivi d'une explication, et chaque graphe d'une **interprétation calculée automatiquement à partir des résultats obtenus**.

## Exécution sur Kaggle (recommandé)

1. Kaggle → **Create → New Notebook** → **File → Import Notebook** → importer le `.ipynb`.
2. **Add Input** → rechercher **« COCO 2017 »** → ajouter un dataset contenant `train2017`, `val2017` et `annotations` (le notebook localise automatiquement les dossiers).
3. **Settings → Accelerator → GPU** (T4 ou P100) et **Internet → On**.
4. **Save Version → Save & Run All**.

Configuration par défaut : 6 000 images de train2017, 500 images de val2017, 6 époques, images 512×853 (≈ 1 h 30 – 2 h sur GPU). Ces valeurs se modifient dans le dictionnaire `CONFIG`.

Les résultats (modèle `.pth`, historique, figures, `recap.json`) sont enregistrés dans `/kaggle/working/outputs`.

## Exécution locale (CPU, test rapide)

```bash
pip install -r requirements.txt
jupyter notebook TP_FasterRCNN_ResNet50_COCO_Kaggle.ipynb
```

Mettre `MODE = "coco128"` : le notebook télécharge automatiquement 128 images de COCO et des poids ResNet-50 ImageNet, et s'exécute sur CPU (performances limitées par le faible nombre d'images).

## Auteur

Ismail En-niou
