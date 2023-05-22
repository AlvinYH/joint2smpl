from __future__ import print_function, division
import argparse
import torch
import os,sys
from os import walk, listdir
from os.path import isfile, join
import numpy as np
import pickle
import smplx
import trimesh
import h5py

sys.path.append(os.path.join(os.path.dirname(__file__), "src"))
from smplify import SMPLify3D
import config

def fit_smpl(data, dir_save, idx_person):
	num_frames = data.shape[0]
	num_space = 5  # visualize one in every 5 frames
	for idx in range(0, num_frames, num_space):
		joints3d = data[idx]
		keypoints_3d[0, :, :] = torch.Tensor(joints3d).to(device).float()

		if idx == 0:
			pred_betas[0, :] = init_mean_shape
			pred_pose[0, :] = init_mean_pose
			pred_cam_t[0, :] = cam_trans_zero
		else:
			curr_path = os.path.join(dir_save, "frame_{:04d}_person_{}.pkl".format(idx-num_space, idx_person))
			with open(curr_path, 'rb') as fr:
				data_param = pickle.load(fr)

			pred_betas[0, :] = torch.from_numpy(data_param['beta']).unsqueeze(0).float()
			pred_pose[0, :] = torch.from_numpy(data_param['pose']).unsqueeze(0).float()
			pred_cam_t[0, :] = torch.from_numpy(data_param['cam']).unsqueeze(0).float()
			
		if opt.joint_category =="AMASS":
			confidence_input =  torch.ones(opt.num_joints)
			# make sure the foot and ankle
			if opt.fix_foot == True:
				confidence_input[7] = 1.5
				confidence_input[8] = 1.5
				confidence_input[10] = 1.5
				confidence_input[11] = 1.5
		else:
			print("Such category not settle down!")
		
		# ----- from initial to fitting -------
		new_opt_vertices, new_opt_joints, new_opt_pose, new_opt_betas, \
		new_opt_cam_t, new_opt_joint_loss = smplify(
													pred_pose.detach(),
													pred_betas.detach(),
													pred_cam_t.detach(),
													keypoints_3d,
													conf_3d=confidence_input.to(device),
													seq_ind=idx
													)

		# # -- save the results to ply---
		outputp = smpl_model(betas=new_opt_betas, global_orient=new_opt_pose[:, :3], body_pose=new_opt_pose[:, 3:],
							transl=new_opt_cam_t, return_verts=True)
		mesh_p = trimesh.Trimesh(vertices=outputp.vertices.detach().cpu().numpy().squeeze(), faces=smpl_model.faces, process=False)
		mesh_p.export(os.path.join(dir_save, "frame_{:04d}_person_{}.ply".format(idx, idx_person)))
		
		# save the pkl
		param = {}
		param['beta'] = new_opt_betas.detach().cpu().numpy()
		param['pose'] = new_opt_pose.detach().cpu().numpy()
		param['cam'] = new_opt_cam_t.detach().cpu().numpy()
		with open(os.path.join(dir_save, "frame_{:04d}_person_{}.pkl".format(idx, idx_person)), 'wb') as fw:
			pickle.dump(param, fw)


# parsing argmument
parser = argparse.ArgumentParser()
parser.add_argument('--batchSize', type=int, default=1,
                    help='input batch size')
parser.add_argument('--num_smplify_iters', type=int, default=100,
                    help='num of smplify iters')
parser.add_argument('--cuda', type=bool, default=True,
                    help='enables cuda')
parser.add_argument('--gpu_ids', type=int, default=0,
                    help='choose gpu ids')
parser.add_argument('--num_joints', type=int, default=15,
                    help='joint number')
parser.add_argument('--joint_category', type=str, default="AMASS",
                    help='use correspondence')
parser.add_argument('--fix_foot', type=str, default="False",
                    help='fix foot or not')
parser.add_argument('--data_folder', type=str, default="./data",
                    help='data in the folder')
parser.add_argument('--save_folder', type=str, default="./results",
                    help='results save folder')
parser.add_argument('--files', type=str, default="1009_5:31.npy",
                    help='files use')
opt = parser.parse_args()
print(opt)

# ---load predefined something
device = torch.device("cuda:" + str(opt.gpu_ids) if opt.cuda else "cpu")
smpl_model = smplx.create(config.SMPL_MODEL_DIR, 
                         model_type="smpl", gender="neutral", ext="pkl",
                         batch_size=opt.batchSize).to(device)

# ## --- load the mean pose as original ---- 
smpl_mean_file = config.SMPL_MEAN_FILE

file = h5py.File(smpl_mean_file, 'r')
init_mean_pose = torch.from_numpy(file['pose'][:]).unsqueeze(0).float()
init_mean_shape = torch.from_numpy(file['shape'][:]).unsqueeze(0).float()
cam_trans_zero = torch.Tensor([0.0, 0.0, 0.0]).to(device)
#
pred_pose = torch.zeros(opt.batchSize, 72).to(device)
pred_betas = torch.zeros(opt.batchSize, 10).to(device)
pred_cam_t = torch.zeros(opt.batchSize, 3).to(device)
keypoints_3d = torch.zeros(opt.batchSize, opt.num_joints, 3).to(device)

# # #-------------initialize SMPLify
smplify = SMPLify3D(smplxmodel=smpl_model,
                    batch_size=opt.batchSize,
                    joints_category=opt.joint_category,
					num_iters=opt.num_smplify_iters,
                    device=device)

# --- load data ---
purename = os.path.splitext(opt.files)[0]
data = np.load(opt.data_folder + "/" + purename + ".npy")

dir_save = os.path.join(opt.save_folder, purename)
if not os.path.isdir(dir_save):
	os.makedirs(dir_save, exist_ok=True) 

num_persons, num_frames,  _, _ = data.shape
# data = data.reshape(num_segs, num_persons, num_frames, 15, 3)
for j in range(num_persons):
	# curr_dir_save = os.path.join(dir_save, "seg_{:04d}".format(i))
	if not os.path.isdir(dir_save):
		os.makedirs(dir_save, exist_ok=True) 
	fit_smpl(data[j], dir_save, j)