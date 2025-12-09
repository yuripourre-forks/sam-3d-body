conda activate sam_3d_body

python main.py     --image_folder /home/supergamebuilder/images     --output_folder /home/supergamebuilder/images/output     --checkpoint_path ./checkpoints/sam-3d-body-dinov3/model.ckpt     --mhr_path ./checkpoints/sam-3d-body-dinov3/assets/mhr_model.pt

#python main.py     --export_bvh --left_handed --image_folder /home/supergamebuilder/images     --output_folder /home/supergamebuilder/images/output     --checkpoint_path ./checkpoints/sam-3d-body-dinov3/model.ckpt     --mhr_path ./checkpoints/sam-3d-body-dinov3/assets/mhr_model.pt
