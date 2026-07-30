# Dataset — Rotten Mango vs Formalin-Mixed Mango

## Source

**Kaggle:** [Fruits Disease Dataset](https://www.kaggle.com/datasets/saravanansri/fruits-disease-dataset)
by **Saravanan Sri**

## Brief Description

The dataset contains RGB images of multiple fruit types (Apple, Banana, Grape, Mango,
Orange) each categorised into condition classes (Fresh, Formalin-mixed, Rotten). For
this project only the **Mango** sub-folder is used, and only the **Rotten** and
**Formalin-mixed** classes are selected — Fresh and all other fruits are excluded.
Images are pre-split across `train`, `valid`, and `test` directories. The `valid` split
is renamed to `val` for consistency. All images are resized to **224 × 224** pixels
during loading. For full details on collection method and licensing refer to the dataset
page linked above.

## Structure (used subset)

```
Dataset/
├── train/  Mango/  ├── Rotten/
│                   └── Formalin-mixed/
├── valid/  Mango/  ├── Rotten/
│                   └── Formalin-mixed/
└── test/   Mango/  ├── Rotten/
                    └── Formalin-mixed/
```
