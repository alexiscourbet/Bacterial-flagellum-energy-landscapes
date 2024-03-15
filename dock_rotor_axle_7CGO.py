#!/usr/bin/python
import os
import numpy as np
import pyrosetta
import imp
from math import *
#from xyzMath import *
from rif.legacy.xyzMath import *
import repeat_utils_w_contact_list 
import hash_subclass_universal


def dock_rotor(p,up_or_down,rot_bins):
    for direction in up_or_down:
      #p_uod=repeat_utils_w_contact_list.center_com_z(p.clone())
      p_up_or_down=repeat_utils_w_contact_list.rotate_around_x(Vec(1.,0.,0.),direction,p.clone())
      for deg in rot_bins:
        print ("Sampling combination",direction, deg)
        c=repeat_utils_w_contact_list.rotate_around_z(Vec(0.,0.,1.0),deg,p_up_or_down.clone())
        yield(c,direction,deg)

args={}

#############################################

#init Rosetta
pyrosetta.init()

pose_axle_path="axle_aligned_trunc.pdb"
pose_axle=pyrosetta.pose_from_pdb(pose_axle_path)
pose_axle_centeredx=repeat_utils_w_contact_list.center_com_x(pose_axle.clone())
pose_axle_centeredxy=repeat_utils_w_contact_list.center_com_y(pose_axle_centeredx.clone())
pose_rotor=pyrosetta.pose_from_pdb("rotor_aligned_trunc.pdb")
pose_rotor_centeredx=repeat_utils_w_contact_list.center_com_x(pose_rotor.clone())
pose_rotor_centeredxy=repeat_utils_w_contact_list.center_com_y(pose_rotor_centeredx.clone())

print ("Sampling according to dock params")
dock_gen=dock_rotor(pose_rotor_centeredxy,np.arange(0.,360,360),np.arange(0.0,360.1,0.1))

print ("Outputting pdbs...")

for ipose,direct,degree in dock_gen:
    combined_rotor_clone=pose_axle_centeredxy.clone()
    combined_rotor_clone.append_pose_by_jump(ipose.clone(),1)
    combined_rotor_clone.dump_pdb('sampled_%s_%s.pdb'%(direct,degree))

print ("Done")


