import os
import matplotlib.pyplot as plt
from pycocotools.coco import COCO
import random
import cv2
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


def filter_images(dataset, min_size, categories):
    filt_image_ids = {}
    cat_ids = dataset.getCatIds(catNms=categories)
    image_ids = dataset.getImgIds(catIds=cat_ids)
    assert(len(image_ids) == len(set(image_ids)))

    for img_id in image_ids:
        ann_ids = dataset.getAnnIds(imgIds=img_id)
        anns = dataset.loadAnns(ann_ids)
        for ann in anns:
            if ann['category_id'] not in cat_ids:
                continue

            bbox = ann['bbox']
            cat_idx = cat_ids.index(ann['category_id'])
            cat = categories[cat_idx]
            
            if (bbox[2] >= min_size) and (bbox[3] >= min_size):
                if cat not in filt_image_ids:
                    filt_image_ids[cat] = []
                filt_image_ids[cat].append({
                    'image_id': img_id,
                    'bbox': bbox
                })
    
    return filt_image_ids


def write_images(dataset, filt_img_ids):
    for cat in list(filt_img_ids.keys()):
        counter = 0
        save_path = os.path.join(args.data_path, '{}_crop'.format(args.version), cat)
        os.makedirs(os.path.join(save_path), exist_ok=True)
        img_ids = filt_img_ids[cat]
        pbar = tqdm(total=len(img_ids), desc='Writing images for {}'.format(cat))
        for img_id in img_ids:
            img_info = dataset.loadImgs(img_id['image_id'])[0]
            img_path = os.path.join(args.data_path, args.version, img_info['file_name'])
            img = cv2.imread(img_path)
            x, y, w, h = img_id['bbox']
            x, y, w, h = int(x), int(y), int(w), int(h)
            sub_img = img[y:y+h, x:x+w]
            sub_img = cv2.resize(sub_img, (args.min_size, (args.min_size * sub_img.shape[0]) // sub_img.shape[1]))
            cv2.imwrite(os.path.join(save_path, str(counter) + '.jpg'), sub_img)
            counter += 1
            pbar.update(1)
        pbar.close()


if __name__ == "__main__":
    categories = ["person", "car"]
    dataset = load_data(args)
    filt_img_ids = filter_images(dataset, args.min_size, categories)
    write_images(dataset, filt_img_ids)
    
    
