import os
import cv2
import json
import re
import datetime
import numpy as np

id_to_category = {}
category_to_id = {}

def pre_process_annos(path):
    thing_classes = []
    id = 0
    with open(path) as f:
        anno_dict = json.load(f)
        for key in anno_dict.keys():
            annos = anno_dict[key]['regions']
            for idx, anno in enumerate(annos):
                category = anno['region_attributes']['name']
                category = category.lower()
                category = re.sub(r"\s+", "", category, flags=re.UNICODE)

                if category not in category_to_id.keys():
                    category_to_id[category] = id
                    id_to_category[id] = category
                    id = id + 1

                    thing_classes.append(category)

                # fix annotations to be the same
                anno_dict[key]['regions'][idx]['region_attributes']['name'] = category
    
    # write the corrected annotation to the old annotation file
    with open(path, "w") as out:
        out.write(json.dumps(anno_dict, separators=(',', ':')))
    out.close()
    return thing_classes

def create_image_info(image_id, file_name, image_size,
                      date_captured=datetime.datetime.utcnow().isoformat(' '),
                      license_id=1, coco_url="", flickr_url=""):
    image_info = {
            "id": image_id,
            "file_name": file_name,
            "width": image_size[0],
            "height": image_size[1],
            "date_captured": date_captured,
            "license": license_id,
            "coco_url": coco_url,
            "flickr_url": flickr_url
    }
    return image_info

def create_annotation_info(annotation_id, image_id, category_id, is_crowd,
                           area, bounding_box, segmentation):
    annotation_info = {
        "id": annotation_id,
        "image_id": image_id,
        "category_id": category_id,
        "iscrowd": is_crowd,
        "area": area,  # float
        "bbox": bounding_box,  # [x,y,width,height]
        "segmentation": segmentation  # [polygon]
    }

    return annotation_info

def get_segmentation(px, py):
    segm = []
    for x,y in zip(px, py):
        segm.append(x)
        segm.append(y)
    return segm

def get_polygon_area(px, py):
    return 0.5*np.abs(np.dot(px,np.roll(py,1))-np.dot(py,np.roll(px,1)))

class NumpyEncoder(json.JSONEncoder):
    """ Custom encoder for numpy data types """
    def default(self, obj):
        if isinstance(obj, (np.int_, np.intc, np.intp, np.int8,
                            np.int16, np.int32, np.int64, np.uint8,
                            np.uint16, np.uint32, np.uint64)):

            return int(obj)

        elif isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
            return float(obj)

        elif isinstance(obj, (np.complex_, np.complex64, np.complex128)):
            return {'real': obj.real, 'imag': obj.imag}

        elif isinstance(obj, (np.ndarray,)):
            return obj.tolist()

        elif isinstance(obj, (np.bool_)):
            return bool(obj)

        elif isinstance(obj, (np.void)): 
            return None

        return json.JSONEncoder.default(self, obj)

def via2coco(img_path, anno_path, info):
    coco_output = {}    
    coco_output['info'] = {
        'description': info['data_description'],
        'url': info['url'],
        'version': info['version'],
        'year': info['year'],
        "contributor": info['contributor'],
        "date_created": datetime.datetime.utcnow().isoformat(' ')
    }

    # register all the categories
    thing_classes = pre_process_annos(anno_path)
    coco_output['categories'] = []
    for id in id_to_category.keys():
        temp = {
            'id': id,
            'name': id_to_category[id],
        }
        coco_output['categories'].append(temp)

    # process images & annotations
    coco_output['images'] = []
    coco_output['annotations'] = []

    annos = json.load(open(anno_path))
    anno_id = 0

    for img_id, key in enumerate(annos.keys()):
        file_name = annos[key]['filename']
        img = cv2.imread(os.path.join(img_path, file_name))

        # make image info and storage it in coco_output['images']
        image_info = create_image_info(img_id, os.path.basename(file_name), img.shape[:2])
        coco_output['images'].append(image_info)

        # within 1 image, there can be multiple regions/instances (which all share the same img_id)
        regions = annos[key]["regions"]
        for region in regions:
            class_name = region['region_attributes']['name']
            assert(class_name in thing_classes)

            px = region['shape_attributes']['all_points_x']
            py = region['shape_attributes']['all_points_y']
            area = get_polygon_area(px, py)

            min_x = np.min(px)
            max_x = np.max(px)
            min_y = np.min(py)
            max_y = np.max(py)

            bbox = [min_x, min_y, max_x - min_x, max_y - min_y]
            segm = get_segmentation(px, py)
            is_crowd = 0

            annos_info = create_annotation_info(anno_id, img_id, category_to_id[class_name], is_crowd, area, bbox, segm)
            coco_output['annotations'].append(annos_info)
            anno_id = anno_id + 1
    
    return coco_output



if __name__ == '__main__':
    print('Hello World')

    info = {
        'data_description': 'Roadstress Dataset',
        'url': 'https//github.com/KossBoii/detr',
        'version': '0.0.1',
        'year': '2022',
        'contributor': 'Long Truong',
    }
    img_path = os.path.join(os.getcwd(), 'dataset', 'roadstress')
    anno_path = os.path.join(img_path, 'via_export_json.json')
    
    # convert annotations from VIA format to COCO format
    coco_train_annos = via2coco(img_path, anno_path, info)

    # save COCO annotations
    with open(os.path.join(img_path, 'coco_train_annos.json'), 'w', encoding="utf-8") as outfile:
        json.dump(coco_train_annos, outfile, sort_keys=True, indent=4, ensure_ascii=False)

