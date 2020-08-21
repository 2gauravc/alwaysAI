import logging
import time
import edgeiq
import numpy as np

"""
Use pose estimation to determine human poses in realtime. Human Pose returns
a list of key points indicating joints that can be used for applications such
as activity recognition and augmented reality.

Pose estimation is only supported using the edgeIQ container with an NCS
accelerator.

To install app dependencies in the runtime container, list them in the requirements.txt file.
"""

def arr_to_img(frame,num):
    from PIL import Image
    import numpy as np

    w, h = 512, 512
    data = np.zeros((h, w, 3), dtype=np.uint8)
    data[0:256, 0:256] = [255, 0, 0] # red patch in upper left
    img = Image.fromarray(frame, 'RGB')
    img.save('my.png')

def main():
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
            time.sleep(2.0)
            fps.start()
            i = 0
            # loop detection
            while True:
                
                frame = video_stream.read()
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
                    if (i ==0):
                        print (pose.key_points)
                    for key_point in pose.key_points:
                        text.append(str(key_point))
                #print(results.raw_results)
    
                #streamer.send_data(results.draw_poses(frame), text)

                fps.update()
                i+=1

                # write the image
                arr_to_img(frame, i)
                
                if streamer.check_exit():
                    break
    finally:
        fps.stop()
        print("elapsed time: {:.2f}".format(fps.get_elapsed_seconds()))
        print("approx. FPS: {:.2f}".format(fps.compute_fps()))

        print("Program Ending")


if __name__ == "__main__":
    main()
