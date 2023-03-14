raw_data_dir=$1
data_dir_2hz=$2
data_dir_20hz=$3
version=$4
split=$5

# token information
python token_info.py --raw_data_folder $raw_data_dir --data_folder $data_dir_2hz --mode 2hz --version $version --split $split
python token_info.py --raw_data_folder $raw_data_dir --data_folder $data_dir_20hz --mode 20hz --version $version --split $split

# time stamp information
python time_stamp.py --raw_data_folder $raw_data_dir --data_folder $data_dir_2hz --mode 2hz --version $version --split $split
python time_stamp.py --raw_data_folder $raw_data_dir --data_folder $data_dir_20hz --mode 20hz --version $version --split $split

# sensor calibration information
python sensor_calibration.py --raw_data_folder $raw_data_dir --data_folder $data_dir_2hz --mode 2hz --version $version --split $split
python sensor_calibration.py --raw_data_folder $raw_data_dir --data_folder $data_dir_20hz --mode 20hz --version $version --split $split

# ego pose
python ego_pose.py --raw_data_folder $raw_data_dir --data_folder $data_dir_2hz --mode 2hz --version $version --split $split
python ego_pose.py --raw_data_folder $raw_data_dir --data_folder $data_dir_20hz --mode 20hz --version $version --split $split

# gt information
python gt_info.py --raw_data_folder $raw_data_dir --data_folder $data_dir_2hz --mode 2hz --version $version --split $split
python gt_info.py --raw_data_folder $raw_data_dir --data_folder $data_dir_20hz --mode 20hz --version $version --split $split

# point cloud, useful for visualization
#python raw_pc.py --raw_data_folder $raw_data_dir --data_folder $data_dir_2hz --mode 2hz
#python raw_pc.py --raw_data_folder $raw_data_dir --data_folder $data_dir_20hz --mode 20hz

