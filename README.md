# DriViSafe

Example for running SimpleViT model:
```console
python drivisafe/mpl2/main.py \
--name=SimpleViT \
--dataset=dreyeve  \
--num-classes=2 \
--batch-size=1 \
--eval-step=500 \
--num_train_lb=200 \
--num_val=16 \
--workers=10 \
--model=simplevit
```

To increase limit of open files simultaneously:
```console
ulimit -n 100000
```

Label Studio init from terminal:
```console
LABEL_STUDIO_LOCAL_FILES_SERVING_ENABLED=true \
LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT=/media/alberto/Elements/Datasets/ \
label-studio start --sampling uniform
```

Absolute path config in Label Studio:
```console
/mnt/d/Datasets/Dr(eye)ve
```

Update labels in the targets folder:
```
rm -r ./* && aws s3 sync s3://dreyeve-targets-cars-people ./ && clear && ls -l && ls -1 | wc -l
```
