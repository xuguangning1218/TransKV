# TransKV

Official source code for the paper *“TransKV: A Data-Driven Pruning Method for Large Foundation Models”*, accepted to **CVPR 2026 Findings**.

---

## 🔥 Highlights

* 🔍 **Data-driven KV pruning** without retraining
* ⚡ Compatible with **LLaMA, ViT, Whisper, Qwen, DeepSeek-OCR**
* 📉 Achieves up to **50% pruning ratio** with minimal performance drop
* 🧠 Supports both **vision, speech, and language foundation models**

---

## 🏗️ Overall Architecture

![TransKV](https://github.com/xuguangning1218/TransKV/blob/master/figures/TransKV.png)
---

## ⚙️ Environment Setup

```bash
git clone git@github.com:xuguangning1218/TransKV.git
cd TransKV

conda create -n transkv python==3.11.13 -y && conda activate transkv

pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu128

pip install -r requirements.txt
```

---

## 📂 Data Preparation

* **Image Task**
  Prepare the ImageNet-1K dataset from the
  https://www.image-net.org/download.php

* **Speech Task**
  Prepare the Common Voice 17.0 dataset from
  https://datacollective.mozillafoundation.org/organization/cmfh0j9o10006ns07jq45h7xk
  ⚠️ *Note: This dataset is no longer available on Hugging Face.*

* **Text Task**
  The following datasets will be automatically downloaded via Hugging Face:

  * ARC-Challenge
  * ARC-Easy
  * BoolQ
  * HellaSwag
  * OpenBookQA
  * PIQA
  * Winogrande

---

## 🤖 Supported Models

* **Image Models**

  * ViT-L
  * **ViT-B** (provided)
  * ViT-Tiny

* **Speech Models**

  * **Whisper-small** (provided)

* **Language Models**

  * **LLaMA3-8B** (provided)
  * Qwen2.5 (0.5B / 1.5B / 3B / 7B / 14B)
  * SmolLM3

* **Others**

  * DeepSeek-OCR

---

## 📁 Project Structure

```bash
TransKV/
├── figure/                  # figures and tables
│   └── TransKV.png         # architecture
├── model/                  # core pruning code
│   ├── LLama3/
│   │   ├── extract_pca.py  # obtain PCA matrices
│   │   ├── test_pca.py     # evaluate PCA matrices
│   │   ├── transkv.py      # core pruning implementation
│   │   ├── test_LLama3.py  # evaluation scripts
│   │   └── modeling_llama.py # non-invasive transformer modification
│   ├── ViT/                # ViT-B support
│   └── Whisper-small/      # Whisper support
├── pca_matrices/           # PCA storage
│   ├── LLama3/
│   ├── ViT/
│   └── Whisper-small/
├── results/                # experiment results
│   ├── meta-llama__Meta-Llama-3-8B # original results
│   └── transkv_LLama3      # 50% pruning results
├── scripts/
│   ├── llama.sh
│   ├── transkv_llama.sh
│   ├── transkv_vit.sh
│   └── transkv_whisper.sh
```

---

## 🚀 Running

Before running, please:

* Update `ROOT_DIR` in scripts
* Update your conda activation path

```bash
cd TransKV
bash ./scripts/transkv_llama.sh
```

---

## 📊 Results

Example results for **LLaMA3-8B (50% pruning ratio)** are available in:

```
results/transkv_LLama3
```

---

## 📌 TODO

* [ ] Support vision-language model
* [ ] Add more model support (e.g., Mistral, Gemma)

---

## 📖 Citation

```bibtex
@InProceedings{Xu_2026_CVPR,
    author    = {Xu, Guangning and Meng, Fanxu and Zhou, Ruijie and Ng, Michael K and Pei, Wenjie and Zhang, Muhan},
    title     = {TransKV: A Data-Driven Pruning Method for Large Foundation Models},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) Findings},
    month     = {June},
    year      = {2026},
    pages     = {2451-2461}
}
```

---

## ⭐ Acknowledgement

If you find this work useful, please consider giving a ⭐ to the repository.
