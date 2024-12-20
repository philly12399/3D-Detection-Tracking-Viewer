from viewer.viewer import Viewer
import numpy as np
from dataset.kitti_dataset import KittiTrackingDataset
import os
import click
from tqdm import tqdm

@click.command()
### Add your options here
@click.option(
    "--data_root",
    "-d",
    type=str,
    default="/home/philly12399/philly_ssd/KITTI_tracking/training/",
    help="Path of kitti-track dataset root",
)
@click.option(
    "--seq",
    "-s",
    type=int ,
    default="0",

    help="Path of sv episodes",
)
@click.option(
    "--box_type",
    "-b",
    type=str ,
    default="Kitti",
    help="bbox format {Kitti, OpenPCDet, Waymo, Philly}",
)
@click.option(
    "--label_root",
    "-r",
    type=str ,
    default="",
    help="Path of label root",
)
@click.option(
    "--label_name",
    "-l",
    type=str ,
    default="",
    help="Path of label",
)
@click.option(
    "--color_by_cls",
    "-c",
    type=bool ,
    default=False,
    help="color_by_cls(true) or color by trkid(false)",
)
@click.option(
    "--start",
    "-start",
    type=int ,
    default=0,
    help="start index",
)
@click.option(
    "--ids",
    "-ids",
    type=bool ,
    default=False,
    help="show ids only",
)
def kitti_viewer(data_root, seq, box_type, label_root, label_name,color_by_cls,start,ids):
    # root="/home/philly12399/nas/homes/arthur_data/KITTI_tracking/training/"
    # label_path = r"/home/philly12399/nas/homes/arthur_data/KITTI_tracking/training/label_02/0001.txt"
    # data_root = "/home/philly12399/philly_data/pingtung-tracking-val/val/kitti-format/tracktest/"
    if(label_root == ""):
        label_root = os.path.join(data_root,"label_02")
    if(label_name == ""):
        label_name = str(seq).zfill(4)+".txt"
    label_path = os.path.join(label_root, label_name)
    # COLOR_BY_CLS = False
    # label_path = os.path.join(data_root, "label_02/" + "track.txt")
    
    dataset = KittiTrackingDataset(data_root, seq_id=seq, box_type = box_type, label_path=label_path)

    vi = Viewer(box_type= box_type)
    if(ids):
        # ids_list = [(10, 11, 0), (13, 15, 20), (18, 21, 20), (20, 26, 8), (28, 36, 24), (32, 47, 10), (33, 40, 8), (34, 43, 32), (56, 59, 16), (61, 63, 11),(73, 80, 0), (80, 73, 0), (76, 83, 0), (83, 76, 0), (77, 93, 0), (93, 77, 0), (78, 87, 0), (87, 78, 0), (78, 96, 0), (96, 78, 0), (79, 86, 12), (86, 89, 0), (89, 86, 0), (82, 94, 0), (94, 82, 0), (84, 90, 0), (90, 84, 0), (84, 99, 0), (99, 84, 0), (85, 88, 0), (88, 85, 0), (85, 98, 0), (98, 85, 0), (92, 95, 0), (95, 92, 0)]
        # id_list=[]
        # for i in ids_list:
        #     id_list.append(i[0])
        #     id_list.append(i[1])
        id_list=16
            
    # for i in tqdm(range(start,len(dataset))):
    for i in range(start,len(dataset)):    
        print("Frame: ",i)
        P2, V2C, points, image, labels, label_names = dataset[i]
        cls_list = ["Car","Cyclist","FilteredCar","FilteredCyclist","KF_Car","KF_Cyclist","Van"]
        color_list = {"Car":[0,0,255],"Cyclist":[0,255,0],"FilteredCar":[255,0,0],"FilteredCyclist":[255,0,0],"KF_Car":[255,0,0],"KF_Cyclist":[255,0,0],}
        # cls_list = ["Cyclist"]
        
        if labels is not None:           
            # mask = (label_names!="DontCare")
            mask = np.isin(label_names, cls_list)            
            labels = labels[mask]
            label_names = label_names[mask]
            if(ids):
                label_ids = labels[:, -1].astype(int)
                mask = np.isin(label_ids, id_list)
                labels = labels[mask]
                label_names = label_names[mask]

            if(color_by_cls):
                colors = [color_list[label_names[i]] for i in range(len(label_names))]
                vi.add_3D_boxes(labels, ids=labels[:, -1].astype(int), box_info=label_names,caption_size=(0.05,0.05),my_color=colors)
            else:
                vi.add_3D_boxes(labels, ids=labels[:, -1].astype(int), box_info=label_names,caption_size=(0.05,0.05))
            # vi.add_3D_cars(labels, ids=labels[:, -1].astype(int), mesh_alpha=1)
        vi.add_points(points[:,:3])
        
        try:
            vi.add_image(image)
        except:
            pass
        vi.set_extrinsic_mat(V2C)
        vi.set_intrinsic_mat(P2)

        # vi.show_2D()

        vi.show_3D()
def main():
    kitti_viewer()
    
if __name__ == '__main__':
    main()
