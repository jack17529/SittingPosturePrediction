# Sitting Posture Prediction

NOTE - I have done it with high sensitivity and low durability.
As it needs to be real time thus I could not able to use Deep Learning, whose results are more accurate and durable but are not real time. I have seen the best of the best projects on it but they all require video upload and lag badly with clever tweaking.

Hence this is the best that could happen, it will strongly depend on the performance of the computer and the distance you are sitting from the camera. Please make sure that the points on the face match the points shown in the Readme.md file otherwise the prediction can be inaccurate. Avoid big movements with head to achieve high accuracy.

I used Opencv and Dlib to make this.I then used the pitch angle to calculate the sitting posture by training the model and then testing it for appropriate values.

Hope you enjoy this.

# Head Pose Estimation
Real-time head pose estimation built with OpenCV and dlib 

<b>2D:</b><br>Using dlib for facial features tracking, modified from http://dlib.net/webcam_face_pose_ex.cpp.html
<br>The algorithm behind it is described in http://www.csc.kth.se/~vahidk/papers/KazemiCVPR14.pdf
<br>It applies cascaded regression trees to predict shape(feature locations) change in every frame.
<br>Splitting nodes of trees are trained in random, greedy, maximizing variance reduction fashion.
<br>The well trained model can be downloaded from http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2 
<br>Training set is based on i-bug 300-W datasets. It's annotation is shown below:<br><br>
![ibug](https://cloud.githubusercontent.com/assets/16308037/24229391/1910e9cc-0fb4-11e7-987b-0fecce2c829e.JPG)
<br><br>
<b>3D:</b><br>To match with 2D image points(facial features) we need their corresponding 3D model points. 
<br>http://aifi.isr.uc.pt/Downloads/OpenGL/glAnthropometric3DModel.cpp provides a similar 3D facial feature model.
<br>It's annotation is shown below:<br><br>
![gl](https://cloud.githubusercontent.com/assets/16308037/24229340/ea8bad94-0fb3-11e7-9e1d-0a2217588ba4.jpg)
<br><br>
Finally, with solvepnp function in OpenCV, we can achieve real-time head pose estimation.
<br><br>

# Running

1. Download the project and extract it.
2. Install the requirements using <br>
```
pip install -r /path/to/requirements.txt
```
3. Run using Python 3 <br>
```
python video_test_shape.py
```
# Special Thanks :)
lincolnhard's head-pose-estimation repository on GitHub. 
