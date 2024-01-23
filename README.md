# DriViSafe

Example for running SimpleViT model:
```console
python drivisafe/mpl2/main.py \
--name=SimpleViT \
--dataset=dreyeve \
--num-classes=2 \
--batch-size=4 \
--eval-step=200 \
--num_train_lb=200 \
--num_val=16 \
--workers=10 \
--model=simplevit
```

Label Studio init from terminal:
```console
LABEL_STUDIO_LOCAL_FILES_SERVING_ENABLED=true \
LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT=/mnt/d/Datasets/ \
label-studio start --sampling uniform
```

Absolute path config in Label Studio:
```console
/mnt/d/Datasets/Dr(eye)ve
```
