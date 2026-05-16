#!/bin/bash
# Download + extract the ImageNet training set from the Kaggle ILSVRC
# competition mirror. Requires the kaggle CLI configured with your API token
# (https://github.com/Kaggle/kaggle-api). For local runs, drop the SBATCH
# directives and just execute the script.
#SBATCH --job-name=download_imagenet
#SBATCH --partition=YOUR_PARTITION
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=500G
#SBATCH --output=logs/touch_%x-%j.out

# conda activate rae_final

kaggle competitions download -c imagenet-object-localization-challenge -p data/


python3 - << 'PY'
import zipfile, os

zip_path = "data/imagenet-object-localization-challenge.zip"
out_dir  = "data"
keep_prefixes = (
    "ILSVRC/Data/",                 # 图片目录（通常都在这下面）
)
keep_ext = (".JPEG", ".jpg", ".jpeg", ".png")  # 按需改

with zipfile.ZipFile(zip_path) as z:
    names = z.namelist()
    kept = 0
    for name in names:
        if name.endswith("/"):
            continue
        if name.startswith(keep_prefixes) and name.lower().endswith(tuple(e.lower() for e in keep_ext)):
            z.extract(name, out_dir)
            kept += 1
print("extracted files:", kept)
PY