import logging
import time
import edgeiq
"""
Use pose estimation to determine human poses in realtime. Human Pose returns
a list of key points indicating joints that can be used for applications such
as activity recognition and augmented reality.

To change the engine and accelerator, follow this guide:
https://alwaysai.co/docs/application_development/changing_the_engine_and_accelerator.html

To install app dependencies in the runtime container, list them in the
requirements.txt file.
"""


def main():
    # Experiment 1. Always AI with video file (instead of webcam)

    pose_estimator = edgeiq.PoseEstimation("alwaysai/human-pose")
    pose_estimator.load(engine=edgeiq.Engine.DNN)

    print("\tLoaded model:\n{}\n".format(pose_estimator.model_id))
    print("\tEngine: {}".format(pose_estimator.engine))
    print("\tAccelerator: {}\n".format(pose_estimator.accelerator))

    fps = edgeiq.FPS()
    frame_num = 1
    
    #Open the input file 
    myvideo = 'input/vjump-s1a.mov'

    try:
        with edgeiq.FileVideoStream(myvideo) as video_stream:
            fps.start()
            # loop detection
            while True:
                frame = video_stream.read()
                results = pose_estimator.estimate(frame)
                
                # Speed Test
                print("\t Processing Frame# ", frame_num)
                
                frame_num+=1
                fps.update()

    except edgeiq.NoMoreFrames:
        print("\tFinished processing {} frames".format(frame_num))

    finally:
        fps.stop()
        print("\t Elapsed time: {:.2f}".format(fps.get_elapsed_seconds()))
        print("\t Approx. FPS: {:.2f}".format(fps.compute_fps()))
        print("\t Program Ending")
    
    
if __name__ == "__main__":
    main()
