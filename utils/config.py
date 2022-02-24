from yaml import safe_load


class Config:

    def __init__(self):
        with open('../config.yaml') as load_file:
            load_dict = safe_load(load_file)
            self._update_dict(**load_dict)

        # Create complete paths
        # Dataset
        self.dataset = self.personal_path + self.dataset_path
        self.dataset_train = self.dataset + 'train'
        self.dataset_test = self.dataset + 'test'
        self.dataset_val = self.dataset + 'val'

        # Labels
        self.labels = self.personal_path + self.labels_path
        self.labels_train = self.labels + 'bdd100k_labels_images_train.json'
        self.labels_val = self.labels + 'bdd100k_labels_images_val.json'

        # Names
        self.names = self.personal_path + self.names_path
        self.names_train = self.names + 'train.txt'
        self.names_test = self.names + 'test.txt'
        self.names_val = self.names + 'val.txt'

        # Semantic
        self.segmentation = self.personal_path + self.segmentation_path
        self.segmentation_classid_train = self.segmentation + 'train/'
        self.segmentation_classid_val = self.segmentation + 'val/'
#         self.segmentation_classid_train = self.segmentation + 'classids/train/'
#         self.segmentation_classid_val = self.segmentation + 'classids/val/'
#         self.segmentation_colormaps_train = self.segmentation + 'colormaps/train/'
#         self.segmentation_colormaps_val = self.segmentation + 'colormaps/val/'

    def _update_dict(self, **entries):
        self.__dict__.update(entries)
