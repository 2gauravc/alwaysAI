---
title: "Pose estimation using alwaysAI toolkit"
---

## Task Complexity 


| Task                            | Required |
|---------------------------------|----------|
| Software Installation           | Yes      |
| Working on Command Line         | Yes      |
| Requires Reading / Editing Code | No       |

Overall Complexity: Low

## About

[alwaysAI](https://alwaysai.co/) provides developers with a simple and flexible way to deliver deep learning computer vision to laptop / desktop as well s other devices (e.g. cameras, drones etc.). 

In this repo we will be using alwaysAI for human pose estimation.

## Set-Up 

We will be using the alwaysAI CLI (Command Line Interface). 


### 1. Set up your development computer 

You need to install the alwaysAI software. 

Follow the instructions on [this](https://www.alwaysai.co/docs/getting_started/development_computer_setup.html) page to install and configure the software.  


### 1. Working with projects 

You need to set-up a project on alwaysAI website and set it up on your laptop / desktop. 

Follow the instructions on [this](https://www.alwaysai.co/docs/getting_started/working_with_projects.html) page. 
- Choose "Human Pose Estimation"" model 
- Before you configure the app (aai app configure), you need to login. The instructions have somehow missed this step.

```
aai user login
```
Use your login and password for the alwaysAI website when prompted. 


## The App

When the app runs, it starts a service on port 5000. Go to port 5000 on your browser. 

With your webcam input, you will see the body keypoints identified and joined by blue lines. 



- 

