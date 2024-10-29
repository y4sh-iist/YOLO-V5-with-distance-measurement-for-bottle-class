Instructions for running the model



1.	Install python on the system from your browser.
2.	Open terminal.
3.	Then install jupyter notebook by typing ‘pip install jupyter notebook’ in terminal.
4.	Type jupyter notebook in the terminal.
5.	This will open jupyter notebook page on your web browser.
6.	From the home menu browse to the execution folder and open the model to run it.
7.	To pass a downloaded image through the algorithm open the “executefromdownload.ipynb”
8.	Copy the address of the image that you want to pass and paste it in the ‘img =’ line under the load test image section.
9.	Shift + enter every cell of the notebook and it should display the result.
10.	For real time detection navigate to ‘RealTimeExecute.ipynb’ from jupyter notebook homepage.
11.	Shift + enter every cell of the notebook it should start the video by default the video capture is from the webcam of laptop, this can be changed by going to the capturing section and changing the number inside the ‘cap=cv2.VideoCapture(x)’, x=0,1 etc.
12.	For exiting the real time detection use the escape key.
13.	For mac it might cause issue after quitting. So, after quitting force quit python.


** To get correct distance measurement results, enter the focal length of the lens of the camera used in the ‘focal_length’ variable under the non maximum suppression section.

** If the object detected is not a bottle the distance measurement values displayed are irrelevant.
