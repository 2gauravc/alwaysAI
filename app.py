import logging
import time
import edgeiq
from scipy import ndimage
import pandas as pd 
import numpy as np
import datetime
import os
import csv
import argparse

"""
To run this file type:

sudo aai app start -- --filepath <<input_video_file_path>>

Use pose estimation to determine human poses in realtime. Human Pose returns
a list of key points indicating joints that can be used for applications such
as activity recognition and augmented reality.

To change the engine and accelerator, follow this guide:
https://alwaysai.co/docs/application_development/changing_the_engine_and_accelerator.html

To install app dependencies in the runtime container, list them in the
requirements.txt file.
"""
def find_person_of_interest(poses):
    max_pose_score = 0 
    num_ind = 0
    max_pose_index = -1
    for ind, pose in enumerate(poses):
        num_ind +=1
        if pose.score > max_pose_score: 
            max_pose_index = ind
            max_pose_score = pose.score 
    return max_pose_index

def flatten_pose_points(pose_points):
    ##  Std_Body_Point  AlwaysAIBodyPtName
    #0  Forehead        Mid(Left Eye, Right Eye)
    #1  Upper Chest     Left Shoulder, Right Shoulder
    #2  Right Shoulder  Right Shoulder 
    #3  Right Elbow     Right Elbow
    #4  Right Wrist     Right Wrist 
    #5  Left Shoulder   Left Shoulder 
    #6  Left Elbow      Left Elbow
    #7  Left Wrist      Left Wrist 
    #8  Navel           
    #9  Right Hip       Right Hip 
    #10 Right Knee      Right Knee 
    #11 Right Ankle     Right Ankle 
    #12 Left Hip        Left Hip 
    #13 Left Knee       Left Knee 
    #14 Left Ankle      Left Ankle 

    #print("Converting pose points to standard schema")
    # pt0 - Forehead 
    try: 
        pt0_forehead_x = (pose_points['Left Eye'].x + pose_points['Right Eye'].x)/2
        pt0_forehead_y = (pose_points['Left Eye'].y + pose_points['Right Eye'].y)/2
    except: 
        pt0_forehead_x = -1
        pt0_forehead_y = -1
    
    # pt1 - Upper Chest
    try: 
        pt1_upperchest_x = (pose_points['Left Shoulder'].x + pose_points['Right Shoulder'].x)/2
        pt1_upperchest_y = (pose_points['Left Shoulder'].y + pose_points['Right Shoulder'].y)/2
    except: 
        pt1_upperchest_x = -1
        pt1_upperchest_y = -1

    # pt2 - Right Shoulder  
    pt2_rightshoulder_x = pose_points['Right Shoulder'].x
    pt2_rightshoulder_y = pose_points['Right Shoulder'].y

    # pt3 - Right Elbow 
    pt3_rightelbow_x = pose_points['Right Elbow'].x
    pt3_rightelbow_y = pose_points['Right Elbow'].y

    #pt4 - Right Wrist 
    pt4_rightwrist_x = pose_points['Right Wrist'].x
    pt4_rightwrist_y = pose_points['Right Wrist'].y

    #pt5 - Left Shoulder 
    pt5_leftshoulder_x = pose_points['Left Shoulder'].x
    pt5_leftshoulder_y = pose_points['Left Shoulder'].y

    # pt6 - Left Elbow 
    pt6_leftelbow_x = pose_points['Left Elbow'].x
    pt6_leftelbow_y = pose_points['Left Elbow'].y

    #pt7 - Left Wrist 
    pt7_leftwrist_x = pose_points['Left Wrist'].x
    pt7_leftwrist_y = pose_points['Left Wrist'].y

    #pt8 - Navel
    pt8_navel_x = -1
    pt8_navel_y = -1

    #pt9 - Right Hip 
    pt9_righthip_x = pose_points['Right Hip'].x
    pt9_righthip_y = pose_points['Right Hip'].y
 
    #pt10 - Right Knee 
    pt10_rightknee_x = pose_points['Right Knee'].x
    pt10_rightknee_y = pose_points['Right Knee'].y

    #pt11 - Right Ankle 
    pt11_rightankle_x = pose_points['Right Ankle'].x
    pt11_rightankle_y = pose_points['Right Ankle'].y

    #pt12 - Left Hip 
    pt12_lefthip_x = pose_points['Left Hip'].x
    pt12_lefthip_y = pose_points['Left Hip'].y
 
    #pt13 - Left Knee 
    pt13_leftknee_x = pose_points['Left Knee'].x
    pt13_leftknee_y = pose_points['Left Knee'].y

    #pt14 - Left Ankle 
    pt14_leftankle_x = pose_points['Left Ankle'].x
    pt14_leftankle_y = pose_points['Left Ankle'].y

    # Create the series 
    data = {'pt0_forehead_x' : pt0_forehead_x, 'pt0_forehead_y' : pt0_forehead_y, 
            'pt1_upperchest_x' : pt1_upperchest_x, 'pt1_upperchest_y' : pt1_upperchest_y, 
            'pt2_rightshoulder_x':pt2_rightshoulder_x, 'pt2_rightshoulder_y':pt2_rightshoulder_y,
            'pt3_rightelbow_x':pt3_rightelbow_x, 'pt3_rightelbow_y':pt3_rightelbow_y,
            'pt4_rightwrist_x':pt4_rightwrist_x, 'pt4_rightwrist_y':pt4_rightwrist_y, 
            'pt5_leftshoulder_x':pt5_leftshoulder_x, 'pt5_leftshoulder_y':pt5_leftshoulder_y, 
            'pt6_leftelbow_x':pt6_leftelbow_x, 'pt6_leftelbow_y':pt6_leftelbow_y, 
            'pt7_leftwrist_x':pt7_leftwrist_x, 'pt7_leftwrist_y':pt7_leftwrist_y,
            'pt8_navel_x':pt8_navel_x, 'pt8_navel_y':pt8_navel_y, 
            'pt9_righthip_x':pt9_righthip_x, 'pt9_righthip_y':pt9_righthip_y,
            'pt10_rightknee_x':pt10_rightknee_x, 'pt10_rightknee_y':pt10_rightknee_y, 
            'pt11_rightankle_x':pt11_rightankle_x, 'pt11_rightankle_y':pt11_rightankle_y, 
            'pt12_lefthip_x':pt12_lefthip_x, 'pt12_lefthip_y':pt12_lefthip_y, 
            'pt13_leftknee_x':pt13_leftknee_x, 'pt13_leftknee_y':pt13_leftknee_y,
            'pt14_leftankle_x':pt14_leftankle_x, 'pt14_leftankle_y':pt14_leftankle_y
            }
    ds = pd.Series(data)
    return ds

def write_proc_details_to_csv(filename, headers, df):

    data_dict = df.to_dict(orient='records')

    if os.path.isfile(filename):
        with open(filename, 'a') as f:
            f_csv = csv.DictWriter(f, headers)
            f_csv.writerows(data_dict)
    else:
        with open(filename, 'w') as f:
            f_csv = csv.DictWriter(f, headers)
            f_csv.writeheader()
            f_csv.writerows(data_dict)
    



def main(myvideo):
    # Experiment 7. Extract key pose points in a flat structure by video frame or image

    pose_estimator = edgeiq.PoseEstimation("alwaysai/human-pose")
    pose_estimator.load(engine=edgeiq.Engine.DNN)

    print("\tLoaded model:\n{}\n".format(pose_estimator.model_id))
    print("\tEngine: {}".format(pose_estimator.engine))
    print("\tAccelerator: {}\n".format(pose_estimator.accelerator))

    fps = edgeiq.FPS()
    frame_num = 0

    
    #Create the empty data frame to hold the flattened pose points 
    cols =['pt0_forehead_x', 'pt0_forehead_y','pt1_upperchest_x', 'pt1_upperchest_y',
            'pt2_rightshoulder_x', 'pt2_rightshoulder_y', 'pt3_rightelbow_x', 'pt3_rightelbow_y',
            'pt4_rightwrist_x', 'pt4_rightwrist_y', 'pt5_leftshoulder_x', 'pt5_leftshoulder_y',
            'pt6_leftelbow_x', 'pt6_leftelbow_y', 'pt7_leftwrist_x', 'pt7_leftwrist_y', 
            'pt8_navel_x', 'pt8_navel_y', 'pt9_righthip_x', 'pt9_righthip_y', 
            'pt10_rightknee_x','pt10_rightknee_y','pt11_rightankle_x','pt11_rightankle_y',
            'pt12_lefthip_x', 'pt12_lefthip_y','pt13_leftknee_x', 'pt13_leftknee_y',
            'pt14_leftankle_x', 'pt14_leftankle_y']
    
    df = pd.DataFrame(columns=cols)
    
    framelist=[]
    conf_scores = []

    try:
        video_stream = edgeiq.FileVideoStream(myvideo).start()
        fps.start()
        # loop detection
        while video_stream.more():
            frame = video_stream.read()
            frame_num+=1
            rotated_frame = ndimage.rotate(frame, 270)
            results = pose_estimator.estimate(rotated_frame)
                
            # Speed Test
            print("Processing Frame# ", frame_num)
                
            # Draw the poses 
            #img_with_poses = results.draw_poses(rotated_frame) 

            
            # Find person of interest 
            max_pose_index = find_person_of_interest(results.poses)
            max_pose_score = results.poses[max_pose_index].score
            num_ind = len(results.poses)


            print("\tFound {} individuals. Max score {}. KPI {}".format(num_ind, max_pose_score,max_pose_index))


            # Flatten the pose points 
            ds = flatten_pose_points(results.poses[max_pose_index].key_points)
            
            #Update the endoutcome data frame and frame number list
            df=df.append(ds, ignore_index=True)
            framelist.append(frame_num)
            conf_scores.append(round(max_pose_score,2))

            #Increment the counts 
            fps.update()
        

    except edgeiq.NoMoreFrames:
        print("\tFinished processing {} frames".format(frame_num))
    except Exception as e:
        print (e)
    finally:
        fps.stop()
        print("\t Elapsed time: {:.2f}".format(fps.get_elapsed_seconds()))
        print("\t Approx. FPS: {:.2f}".format(fps.compute_fps()))
        video_stream.stop()
    
    
    # Post process the pose points data frame 
    time_now = datetime.datetime.now()
    
    #Prepare the VIDEO_POSE_PROC_DETAILS data 
    head, tail = os.path.split(myvideo)
    df['video_file_name'] = tail 
    df['frame'] = framelist
    df['model_name'] = pose_estimator.model_id
    df['processed_on'] = time_now
    df['confidence'] = conf_scores
    #df.set_index('frame', inplace=True)

    headers = [['video_file_name', 'frame', 'model_name', 'processed_on', 'confidence'], cols]
    new_headers = [item for sub_list in headers for item in sub_list]
    
    df = df[new_headers] # Re-order the columns 
    
    # Write the file to a CSV location
    write_proc_details_to_csv('output/VIDEO_POSE_PROC_DETAILS.csv', new_headers, df)


    #Prepare the VIDEO_POSE_PROC_SUMMARY record 
    #video_file_name = tail
    #model_name = pose_estimator.model_id
    #total_frames = frame_num 
    #total_processing_time_secs = fps.get_elapsed_seconds()
    #FPS = fps.compute_fps()
    #processed_on = time_now 

    #write_proc_summary_to_csv('output/VIDEO_POSE_PROC_DETAILS.csv', headers, df_summary)

    print("\t Program Ending")
    
    
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--filepath", required=True, help="Enter the video file")
    args = vars(ap.parse_args())	
    filepath = args["filepath"]

    print("Processing File: {}".format(filepath))
	
    if os.path.isfile(filepath) == False:
        print("File not found. Exiting...")
        sys.exit(1)
    
    main(filepath)
