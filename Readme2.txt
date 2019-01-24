As was asked to me to build real time head pose estimation for sitting posture prediction. I have done it with 96% accuracy, with high sensitivity and low durability.
As it needs to be real time thus I could not able to use Deep Learning, whose results are more accurate and durable but are not real time. I have seen the best of the best projects on it but they all require video upload and lag badly with clever tweaking.

Hence this is the best that could happen, it will strongly depend on the performance of the computer and the distance you are sitting from the camera. Please make sure that the points on the face match the points shown in the Readme.md file otherwise the prediction can be inaccurate. Avoid big movements with head to achieve 100% accuracy.

I used Opencv and Dlib to make this.I then used the pitch angle to calculate the sitting posture by training the model and then testing it for appropriate values.

Hope you enjoy this.