import os
import matplotlib.pyplot as plt
from pycocotools.coco import COCO
import random
import cv2
import shutil
import argparse
from tqdm import tqdm


parser = argparse.ArgumentParser()
parser.add_argument('--data-path', type=str, default='data/coco')
parser.add_argument('--version', type=str, default='val2017')
parser.add_argument('--min-size', type=int, default=128)
args = parser.parse_args()


def load_data(args):
    ann_file = os.path.join(args.data_path, 'annotations/instances_{}.json'.format(args.version))
    coco = COCO(ann_file)
    return coco


def filter_annotations(dataset, min_size, categories):

    flt_annots= {}

    for cat in categories:
        flt_annots[cat] = []
        cat_id = dataset.getCatIds(catNms=cat)[0]
        image_ids = dataset.getImgIds(catIds=cat_id)
        annots = dataset.loadAnns(dataset.getAnnIds(imgIds=image_ids))
        
        for ann in annots:
            if ann['category_id'] == cat_id and ann['bbox'][2] >= min_size and ann['bbox'][3] >= min_size:
                flt_annots[cat].append({
                    'image_id': ann['image_id'],
                    'bbox': ann['bbox']
                })
    
    cat_ids = dataset.getCatIds(catNms=categories)
    other_cat_ids = [c for c in dataset.getCatIds() if c not in cat_ids]
    
    flt_annots['other'] = []
    for cat_id in other_cat_ids:
        image_ids = dataset.getImgIds(catIds=[cat_id])
        for image_id in image_ids:
            flt_annots['other'].append({
                'image_id': image_id,
                'bbox': None
            })

    return flt_annots


def write_images(dataset, filt_img_ids):
    for cat in list(filt_img_ids.keys()):
        counter = 0
        save_path = os.path.join(args.data_path, '{}_crop'.format(args.version), cat)
        if os.path.exists(save_path):
            shutil.rmtree(save_path)
        os.makedirs(os.path.join(save_path), exist_ok=True)
        img_ids = filt_img_ids[cat]
        pbar = tqdm(total=len(img_ids), desc='Writing images for {}'.format(cat))
        for img_id in img_ids:
            img_info = dataset.loadImgs(img_id['image_id'])[0]
            img_path = os.path.join(args.data_path, args.version, img_info['file_name'])
            img = cv2.imread(img_path)
            if img_id['bbox'] is not None:
                x, y, w, h = img_id['bbox']
                x, y, w, h = int(x), int(y), int(w), int(h)
                img = img[y:y+h, x:x+w]
            img = cv2.resize(img, (args.min_size, (args.min_size * img.shape[0]) // img.shape[1]))
            cv2.imwrite(os.path.join(save_path, str(counter) + '.jpg'), img)
            counter += 1
            pbar.update(1)
        pbar.close()


if __name__ == "__main__":
    categories = ["person", "car", "bicycle"]
    dataset = load_data(args)
    filt_img_ids = filter_annotations(dataset, args.min_size, categories)
    write_images(dataset, filt_img_ids)