import numpy as np
import secrets
import json
import os


def hex_gen():
    return secrets.token_hex(16)


def dec_gen(n):
    return np.random.randint(10 ** n)


class Project:
    """ Class for project generation. """

    def __init__(self, ds_labels:np.ndarray, direct:str):
        """ Definition of project attributes:
            :param: ds_labels:
            :param: direct:
        """
        self.ds_labels = ds_labels
        self.direct = direct
        self.ds_pcl = []
        self.pointcloud_list()
        self.create()

    def __len__(self):
        return self.ds_labels.shape[0]

    def pointcloud_list(self):
        id = dec_gen(8)
        for i in range(self.__len__()):
            self.ds_pcl.append([hex_gen(), id + i])

    def key_id_map_writer(self):
        """ Writing function for supervisely key_id_map file.

            :param: ds_labels: List of label lists
            :param: ds_pcl:    List of point cloud [[keys, ids], [keys, ids], ... ]
            :param: direct:    Supervisely project directory
        """
        objs = {}
        figs = {}
        vids = {}
        for labels, pcl in zip(self.ds_labels, self.ds_pcl):
            vids[pcl[0]] = pcl[1]
            for label in labels:
                objs[label.key_id[0]] = label.key_id[2]
                figs[label.key_id[1]] = label.key_id[3]

        data = {"tags": {}, "objects": objs, "figures": figs, "videos": vids}
        with open(self.direct + 'key_id_map.json', 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=4)

    def pcd_figure(self):
        """ Figure dictionary creator for the supervisely point cloud file.

            :param: ds_labels: List of label lists
            :param: ds_pcl:    List of point cloud [[keys, ids], [keys, ids], ... ]
            :param: direct:    Supervisely project directory
        """
        objs = []
        for labels in self.ds_labels:
            for label in labels:
                objs.append({"key": label.key_id[0], "classTitle": "car", "tags": []})
        cont = 0
        labels_path = self.direct + 'ds0/ann'
        if not os.path.exists(labels_path):
            os.makedirs(labels_path )
        for labels, pcl in zip(self.ds_labels, self.ds_pcl):
            self.pcd_label_writer(labels, pcl, labels_path, objs, cont)
            cont += 1

    @staticmethod
    def pcd_label_writer(labels, pcl, labels_path, objects, n):
        """ Writing function for supervisely point cloud file.

            :param: labels:  List of label dictionary objects
            :param: pcl:     List of point clouds ids
            :param: objects: List of dictionary objects from pcd_figure function
            :param: direct:  Supervisely project directory
            :param: n:       Number of point cloud
        """
        figs = []
        vkey = pcl[0]
        for label in labels:
            figs.append({"key": label.key_id[1], "objectKey": label.key_id[0], "geometryType": "cuboid_3d",
                         "geometry": {
                             "position": {"x": label.pos[0], "y": label.pos[1], "z": label.pos[2]},
                             "rotation": {"x": label.rot[0], "y": label.rot[1], "z": label.rot[2]},
                             "dimensions": {"x": label.dim[0], "y": label.dim[1], "z": label.dim[2]}
                         }
                         })
        data = {"description": "", "key": vkey, "tags": [], "objects": objects, "figures": figs}
        pcl_name = str(n).zfill(4)
        with open(labels_path + pcl_name + 'json', 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=4)

    def meta_writer(self):
        """ Writing function for supervisely meta file.

            :param: direct: Supervisely project directory
        """
        id = dec_gen(9)
        cls0 = {"title": "car", "shape": "cuboid_3d", "color": "#F1FA0C", "geometry_config": {}, "id": id, "hotkey": ""}
        cls1 = {"title": "cyclist", "shape": "cuboid_3d", "color": "#50E3C2",
                "geometry_config": {}, "id": id + 1, "hotkey": ""}
        cls2 = {"title": "pedestrian", "shape": "cuboid_3d", "color": "#0CFA2E",
                "geometry_config": {}, "id": id + 2, "hotkey": ""}
        cls3 = {"title": "misc", "shape": "cuboid_3d", "color": "#75E6FF",
                "geometry_config": {}, "id": id + 3, "hotkey": ""}
        classes = [cls0, cls1, cls2, cls3]
        data = {"classes": classes, "tags": [], "projectType": "point_clouds"}
        with open(self.direct + 'meta.json', 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=4)

    def create(self):
        self.meta_writer()
        self.key_id_map_writer()
        self.pcd_figure()
