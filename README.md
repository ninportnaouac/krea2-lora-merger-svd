# Krea2 LoRA Merger & SVD Compressor

A lightweight Windows GUI for merging multiple **Krea 2 LoRAs** into a single `.safetensors` LoRA.

It supports individual LoRA strengths, exact merging, optional **SVD rank compression**, FP16/BF16/FP32 output, and optional trigger metadata.

> This project is an independent community tool. It is not affiliated with or endorsed by Krea.

## Features

- Merge up to 8 Krea 2 LoRAs
- Individual strength for every LoRA
- Exact LoRA merge by factor concatenation
- SVD compression after merge
- Final rank choices: 16 / 32 / 64 / 128 / 256
- FP16 / BF16 / FP32 output
- Trigger metadata field
- Optional trigger `.txt` file
- Compatibility check before merging
- Native Windows Tkinter interface
- No Gradio or browser required
- No Krea 2 base checkpoint required for standard LoRA-to-LoRA merging

## Important limitation

The current LoRA-to-LoRA merger supports **standard LoRA** tensors.

**LoKr is detected but intentionally not merged.** A LoKr update cannot generally be converted into an ordinary low-rank LoRA exactly without an additional approximation/factorization step.

The `Trigger` field stores trigger information in the output metadata and, optionally, in a text file. It does **not** teach a new trigger word to a LoRA. A word only acts as a learned trigger if the source training actually associated that word with the learned concept/style.

## Exact merge vs SVD compression

### Exact mode

The exact mode keeps the complete combined low-rank update. If you merge two rank-32 LoRAs, the resulting layer may become rank 64.

This preserves the merge exactly, but the output file can become larger.

### Compressed SVD mode

The application can compress the merged factors back to a selected target rank.

Example:

```text
LoRA A rank 32
LoRA B rank 32
        ↓
Exact merged rank 64
        ↓
SVD compression
        ↓
Final rank 32 or 64
```

The application displays the average retained SVD energy after compression. Higher retained energy generally means less information was discarded by the rank reduction.

For a first test, **FP16 + rank 64** is a good practical starting point.

## Requirements

- Windows 10/11
- Python 3.11+ recommended
- PyTorch
- safetensors
- Tkinter (included with the standard Windows Python installer)

## Easy installation

Download or clone this repository.

Double-click:

```text
INSTALL.bat
```

This creates a local `venv` and installs the required Python packages.

Then double-click:

```text
RUN_KREA2_LORA_MERGER.bat
```

## Manual installation

```bash
python -m venv venv
venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
python app.py
```

## Usage

1. Click **Ajouter LoRA**.
2. Select your Krea 2 `.safetensors` LoRAs.
3. Select a LoRA in the list and set its weight.
4. Click **Vérifier les LoRA**.
5. Choose:
   - **Exact** for maximum fidelity, or
   - **Compressé SVD** for a smaller output.
6. Choose the final rank if SVD compression is enabled.
7. Choose FP16, BF16, or FP32 output.
8. Optionally enter a trigger word.
9. Click **MERGER LES LoRA**.

## Suggested settings

For a balanced result:

```text
Mode:       Compressed SVD
Rank:       64
Precision:  FP16
```

If the retained SVD energy is too low, try rank 128.

## Output

The application creates a merged `.safetensors` LoRA.

If a trigger is entered and the option is enabled, it also creates:

```text
YourMergedLoRA.txt
```

containing the trigger information.

## Safety and compatibility

Always keep copies of your original LoRAs.

A successful tensor compatibility check means the tensor layouts can be combined by this tool; it does not guarantee that every artistic combination will produce desirable generations.

Before redistributing a merged LoRA, check the licenses and redistribution terms of every source LoRA used in the merge.

## Repository files

```text
krea2-lora-merger-svd/
├── app.py
├── lora_to_lora_merge.py
├── INSTALL.bat
├── RUN_KREA2_LORA_MERGER.bat
├── requirements.txt
├── README.md
├── LICENSE
└── .gitignore
```

## License

MIT License.

---

# Français

## Krea2 LoRA Merger & SVD Compressor

Outil Windows permettant de fusionner plusieurs **LoRA Krea 2** dans un seul fichier `.safetensors`.

### Fonctions principales

- Jusqu'à 8 LoRA
- Poids individuel par LoRA
- Fusion exacte
- Compression SVD optionnelle
- Rank final 16 / 32 / 64 / 128 / 256
- Sortie FP16 / BF16 / FP32
- Champ Trigger
- Fichier `.txt` de trigger optionnel
- Vérification de compatibilité
- Interface Windows Tkinter
- Aucun checkpoint Krea 2 de base nécessaire pour fusionner des LoRA standards

### Installation simple

Double-clique d'abord sur :

```text
INSTALL.bat
```

Puis utilise :

```text
RUN_KREA2_LORA_MERGER.bat
```

### Réglage recommandé pour commencer

```text
Mode :       Compressé SVD
Rank :       64
Précision :  FP16
```

### Attention au Trigger

Le champ Trigger enregistre le mot dans les métadonnées. Il ne crée pas automatiquement un nouveau mot appris. Le trigger doit avoir été associé au concept/style pendant l'entraînement du LoRA d'origine pour avoir un véritable effet sémantique.

### LoKr

Les LoKr sont détectés mais ne sont pas fusionnés par cette version afin d'éviter de produire un fichier incorrect.

### Redistribution

Avant de publier un LoRA fusionné, vérifie les licences des LoRA sources.
