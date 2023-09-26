import torch

class Config:
    voc_labels = ('aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable',
              'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor')
    
    label_map = {k: v + 1 for v, k in enumerate(voc_labels)}
    label_map['background'] = 0


    rev_label_map = {v: k for k, v in label_map.items()}  # Inverse mapping
    
    n_classes = len(voc_labels)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    @classmethod 
    def get_labels(cls) -> tuple[tuple[str], dict[str, int], dict[int, str]]:
        return (cls.voc_labels, cls.label_map, cls.rev_label_map)
    

if __name__ == "__main__":
    print(Config.get_labels())