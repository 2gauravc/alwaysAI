import fastai
from fastai.vision.all import *

"""
Experiment #9 - Import a pickle model into alwaysAI directory and use within app.py

To run this file type:
sudo aai app start 

Use pose estimation to determine human poses in realtime. Human Pose returns
a list of key points indicating joints that can be used for applications such
as activity recognition and augmented reality.

To change the engine and accelerator, follow this guide:
https://alwaysai.co/docs/application_development/changing_the_engine_and_accelerator.html

To install app dependencies in the runtime container, list them in the
requirements.txt file.
"""


def main(myvideo):

    # Print fastai version 
    print('fastai : version {}'.format(fastai.__version__))
    
    # Load the pickle model 
    model_pkl_path = "models/2gauravc/fast_ai_model/efficientnet_lite0__v4.2.pkl"
    try: 
        learn_inf = load_learner(model_pkl_path)
    
    except Exception as e:
        print ("Error: {}".format(e))
    
    finally: 
        print ("All done here")
    
        
    
if __name__ == "__main__":
    main()
