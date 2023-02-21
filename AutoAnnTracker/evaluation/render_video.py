# Make sure ffmpeg is installed in system

import ffmpeg
import argparse
import os
import shutil

def concat_classes(classes, scene_folder, out_name):
    filter_string = "-filter_complex \"[1]format=argb,geq=r='r(X,Y)':a='0.5*alpha(X,Y)'[b];[0:v][b] overlay\""

    concat_dir = os.path.join(os.getcwd(), "concat")
    if os.path.exists(concat_dir):
        shutil.rmtree(concat_dir)
    os.mkdir(concat_dir)

    frame_len = 0
    for c in classes:
        frame_path = os.path.join(scene_folder, c)
        if len(os.listdir(frame_path)) > frame_len:
            frames = os.listdir(frame_path)

    for frame in frames:
        input_string = ""
        for c in classes:
            file_path = os.path.join(scene_folder, c, frame)
            input_string += " -i " + file_path
        out_path = os.path.join(concat_dir, frame)

        shell_command = "ffmpeg " + input_string + " " + filter_string + " " + out_path
        print(shell_command)
        os.system(shell_command)
    
    return concat_dir


def render_video(args):
    print("Rendering video...")
    classes = args.class_names.split(",")
    if len(classes) > 1:
        concat_dir = concat_classes(classes, args.scene_folder, args.out_name)
        file_path = os.path.join(concat_dir,"*.png")
    else:
        file_path = os.path.join(args.scene_folder, args.class_names, "*.png")
    imgs = ffmpeg.input(file_path,  pattern_type='glob', framerate=args.framerate)
    out = ffmpeg.output(imgs, args.out_name)

    out.run()
    print("... Video rendering completed!")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--scene-folder', type=str, help="path to scene")
    parser.add_argument('--class-names', type=str, default = "", help="name of class(es) to render video, e.g. 'car' or 'car,pedestrian'")    
    parser.add_argument('--out-name', type=str, default="movie.mp4", help="name of video")
    parser.add_argument('--framerate', type=int, default=1)
    args = parser.parse_args()
    render_video(args)