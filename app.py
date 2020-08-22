import logging
import time
import edgeiq
import numpy as np
from PIL import Image

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
    img.save('raw_images/run_{}_frm_{}.png'.format(run_id,frame_num))

def main():
    run_id = 1
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

                #Take every 5th frame. Take first 5 instances

                if i % 5 ==0:
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
                        text.append("Key Points:")
                    
                        print ("i={}, j={}\n".format(i,j))
                        print (pose.key_points)

                    for key_point in pose.key_points:
                        text.append(str(key_point))
                    streamer.send_data(results.draw_poses(frame), text)

                    #write the image to disk 
                    arr_to_img(run_id,frame, j)

                    fps.update()
                    j+=1

                i+=1

                                
                if j >=5: #streamer.check_exit():
                    break
    finally:
        fps.stop()
        print("elapsed time: {:.2f}".format(fps.get_elapsed_seconds()))
        print("approx. FPS: {:.2f}".format(fps.compute_fps()))

        print("Program Ending")


if __name__ == "__main__":
    main()
