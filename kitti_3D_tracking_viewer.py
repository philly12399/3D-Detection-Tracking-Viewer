from viewer.viewer import Viewer
import numpy as np
from dataset.kitti_dataset import KittiTrackingDataset
import os
import click
@click.command()
### Add your options here
@click.option(
    "--data_root",
    "-d",
    type=str,
    default="/home/philly12399/philly_data/pingtung-tracking-val/val/kitti-format/tracktest/",
    help="Path of kitti-track dataset root",
)
@click.option(
    "--seq",
    "-s",
    type=int ,
    default="4",

    help="Path of sv episodes",
)
@click.option(
    "--box_type",
    "-b",
    type=str ,
    default="Philly",
    help="bbox format {Kitti, OpenPCDet, Waymo, Philly}",
)
def kitti_viewer(data_root, seq, box_type):
    # root="/home/philly12399/nas/homes/arthur_data/KITTI_tracking/training/"
    # label_path = r"/home/philly12399/nas/homes/arthur_data/KITTI_tracking/training/label_02/0001.txt"
    # data_root = "/home/philly12399/philly_data/pingtung-tracking-val/val/kitti-format/tracktest/"
    label_path = os.path.join(data_root, "label_02/" + str(seq).zfill(4)+".txt")
    # label_path = os.path.join(data_root, "label_02/" + "track.txt")
    
    dataset = KittiTrackingDataset(data_root, seq_id=seq, box_type = box_type, label_path=label_path )

    vi = Viewer(box_type= box_type)

    for i in range(len(dataset)):
        print("Frame: ",i)
        P2, V2C, points, image, labels, label_names = dataset[i]


        if labels is not None:           
            mask = (label_names!="DontCare")
            labels = labels[mask]
            label_names = label_names[mask]
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
