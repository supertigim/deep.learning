## Introduction  

This application explains how Convolutional Neural Nerwork works.  

First, it creates 10x10 image where one dot is located randomly, and the location data divided by height and width is set as a labeled data. Second, CNN is trained width * height * 100 times. Lastly, you check if CNN can predict the position that the dot is located with minimum error rate. Usually, It takes long time to reach an acceptable result, because it needs to do vision processing.     
  
	- Network Input: Image Size 
	- Number of Hidden Layers: 2
	- Netowkr Output: position of dot

임의 위치에 점(또는 십자가)을 찍은 이미지(10x10사이즈)에서 그 점의 위치를 CNN으로 찾아내는 방법으로 동작을 설명. Vision 처리로 원하는 결과를 만들어 내는 것이 CNN의 핵심이며, 덕분에(?) 시간이 많이 걸린다. 1%이하 에러율로 학습하는데 1시간 정도 걸림.     
  
## How to build  
  
	$ mkdir build   
	$ cd build  
  
	build$ cmake ..  
	build$ make  

	build$ ./cnn  
