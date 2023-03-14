raw_data_dir=$1
data_dir=$2
version=$3
split=$4

# token information
python token_info.py --raw_data_folder $raw_data_dir --data_folder $data_dir --split $split --version $version

# time stamp information
python time_stamp.py --raw_data_folder $raw_data_dir --data_folder $data_dir --split $split --version $version

# sensor calibration information
#python sensor_calibration.py --raw_data_folder $raw_data_dir --data_folder $data_dir_2hz --mode 2hz

# ego pose
python ego_pose.py --raw_data_folder $raw_data_dir --data_folder $data_dir --split $split --version $version

#NOT AVALIABLE YET
# gt information
#python gt_info.py --raw_data_folder $raw_data_dir --data_folder $data_dir_2hz --mode 2hz

# point cloud, useful for visualization
#python raw_pc.py --raw_data_folder $raw_data_dir --data_folder $data_dir_2hz --mode 2hz

