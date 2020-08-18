import bin_hdf
import os
import shlex
import subprocess

dataset = '2015-11-09-3/'
subset = 'data000/'
experiment_label = 'experiment03/'
bit = 10
wire = 1
sigma = 5
seed = 100
config = str(bit) + 'b_' + str(wire) + 'w_' + str(sigma) + 'n_0ob_' + str(seed) + 's/'
inp_file =  '/media/guest/data_9TB/jeffrey/data/' + dataset + experiment_label + config + subset + 'mp_data.hdf5'
save_path = '/media/guest/data_9TB/jeffrey/data/' + dataset + experiment_label + config + subset
analysis_path = '/media/guest/data_9TB/jeffrey/analysis/' + dataset + experiment_label + config + subset
if not os.path.exists(analysis_path):
    os.makedirs(analysis_path)

bin_hdf.bin_hdf(inp_file, save_path, 360, 1000, True)

vision_path = '/home/guest/dantemur/utilities/vision7-unix'
vision_rgb = vision_path + '/RGB-8-1-0.48-11111.xml'
pwd = os.getcwd()
os.chdir(vision_path)
subprocess.call(shlex.split('./smash ' + save_path + ' ' + analysis_path + ' ' + vision_rgb))
os.chdir(pwd)
