import json

class Config:
    voc_labels = ('aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable',
              'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor')
    
    label_map = {k: v + 1 for v, k in enumerate(voc_labels)}
    label_map['background'] = 0


    rev_label_map = {v: k for k, v in label_map.items()}  # Inverse mapping
    
    n_classes = len(label_map)

    @classmethod
    def set_labels(cls, DataSetType: str = 'MSCOCO', path_to_labels_json: str = ''):
        if DataSetType.upper() == 'MSCOCO':
            with open(path_to_labels_json) as f:
                data = json.load(f)

            cls.voc_labels = []
            cls.label_map = {}
            cls.rev_label_map = {}
            
            categories = []
            for annotation in data['annotations']:
                categories.append(annotation['category_id'])
            
            for category in data['categories']:
                if category['id'] in categories:
                    cls.voc_labels.append(category['name'])
                    cls.label_map[category['name']] = category['id']
                    cls.rev_label_map[category['id']] = category['name']

            
            cls.label_map['background'] = 0
            cls.rev_label_map[0] = 'background'
            return True
                # cls.label_map[annotation['category_id']] = annotation['id']


            # if isinstance(data, dict):
            #     data = [data]
            
            # cls.label_map = {}
            # cls.rev_label_map = {}
            # voc_labels = []

            # for cat in data:
            #     cls.label_map[cat['name']] = int(cat['id'])
            #     cls.rev_label_map[int(cat['id'])] = cat['name']
            #     voc_labels.append(cat['name'])
            # cls.n_classes = len(cls.voc_labels)
            

            # cls.voc_labels = tuple(voc_labels)
            # return True
        
        return  False
    @classmethod 
    def get_labels(cls) -> tuple[tuple[str], dict[str, int], dict[int, str]]:
        return (cls.voc_labels, cls.label_map, cls.rev_label_map)
    @classmethod
    def get_n_classes(cls):
        return  len(cls.label_map)

