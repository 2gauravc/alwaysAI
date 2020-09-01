import logging
import time
import edgeiq
import numpy as np
from PIL import Image
import math

"""
Use pose estimation to determine human poses in realtime. Human Pose returns
a list of key points indicating joints that can be used for applications such
as activity recognition and augmented reality.

Pose estimation is only supported using the edgeIQ container with an NCS
accelerator.

To install app dependencies in the runtime container, list them in the requirements.txt file.
"""

def arr_to_img(run_id,frame,frame_num):
    img = Image.fromarray(frame, 'RGB')
    img.save('runs/run_{}/images/frm_{}.png'.format(run_id,frame_num))

def calc_dist(p1,p2):

    dist = math.sqrt(pow((p1.x-p2.x),2) + pow ((p1.y-p2.y),2))
    return (dist)

def sanity_check_key_points(key_points):
    #1. Arm lengths are equal
    r_arm_length = calc_dist(key_points['Right Shoulder'], key_points['Right Elbow']) + \
                   calc_dist(key_points['Right Elbow'], key_points['Right Wrist'])

    l_arm_length = calc_dist(key_points['Left Shoulder'], key_points['Left Elbow']) + \
                   calc_dist(key_points['Left Elbow'], key_points['Left Wrist'])

    print("Left Arm: {}".format(l_arm_length))
    print("Right Arm: {}".format(r_arm_length))
    
    #2. Leg lengths are equal
    r_leg_length = calc_dist(key_points['Right Hip'], key_points['Right Knee']) + \
                   calc_dist(key_points['Right Knee'], key_points['Right Ankle'])

    l_leg_length = calc_dist(key_points['Left Hip'], key_points['Left Knee']) + \
                   calc_dist(key_points['Left Knee'], key_points['Left Ankle'])

    l_dict = {'r_arm_length': r_arm_length, 'l_arm_length': l_arm_length, \
              'r_leg_length': r_leg_length, 'l_leg_length':l_leg_length}
    print(l_dict)

    return (l_dict)
    

def main():
    run_id = 4
    pose_estimator = edgeiq.PoseEstimation("alwaysai/human-pose")
    pose_estimator.load(
            engine=edgeiq.Engine.DNN)

    print("Loaded model:\n{}\n".format(pose_estimator.model_id))
    print("Engine: {}".format(pose_estimator.engine))
    print("Accelerator: {}\n".format(pose_estimator.accelerator))

    fps = edgeiq.FPS()

    try:
        with edgeiq.WebcamVideoStream(cam=0) as video_stream, \
                edgeiq.Streamer() as streamer:
            # Allow Webcam to warm up
            time.sleep(5.0)
            fps.start()
            i = 0 # Frame count
            j = 0 # Number of capture count
            # loop detection
            while True:
                
                frame = video_stream.read()

                #Take every 25th frame. Take first 10 instances

                if i % 25 ==0:
                    results = pose_estimator.estimate(frame)
                    # Generate text to display on streamer
                    text = ["Model: {}".format(pose_estimator.model_id)]
                    text.append(
                        "Inference time: {:1.3f} s".format(results.duration))
                    for ind, pose in enumerate(results.poses):
                        text.append("Person {}".format(ind))
                        text.append('-'*10)
                        text.append("Pose Score")
                        text.append(str(pose.score))
                        text.append('-'*10)
                        l_dict=sanity_check_key_points(pose.key_points)

                        text.append("Left Arm Length: {}".format(l_dict['l_arm_length']))
                        text.append("Right Arm Length: {}".format(l_dict['r_arm_length']))
                        text.append("Left Leg Length: {}".format(l_dict['l_leg_length']))
                        text.append("Right Arm Length: {}".format(l_dict['r_leg_length']))

                        print ("i={}, j={}\n".format(i,j))
                        print (pose.score)
    
                    streamer.send_data(results.draw_poses(frame), text)

                    #write the image to disk 
                    arr_to_img(run_id,frame, j)
                    j+=1

                i+=1

                fps.update()
                
                if j >=10: #streamer.check_exit():
                    break
    finally:
        fps.stop()
        print("elapsed time: {:.2f}".format(fps.get_elapsed_seconds()))
        print("approx. FPS: {:.2f}".format(fps.compute_fps()))

        print("Program Ending")


if __name__ == "__main__":
    main()
