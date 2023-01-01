
<h1>MARKATT - The technical overview</h1> 


<h2>WHAT IS IT ABOUT?</h2><br>
What it basically does is very well grasped from the name of the software- MarkATT; standing for “Marking Attendance”
This is the very basic roll out of live detection and record keeping 
The following versions shall aim at using the data collected for data analytics, using the ML pipelines for integrating it in security, and enabling other services useful for an organization.
Hence, while it might seem so, it is much more than a mere attendance system.

<h2>THE TECHNOLOGIES USED AND BROAD ARCHITECTURE</h2><br>
Apart from what other systems work on the scanning-register system, i.e. Scan the available entities- match their coordinates and register their names: MarkATT is based upon a custom, extendable neural network and the product not only offers the client side software but also training, annotation and deploying pipelines which ensure the flexibility to add more people, make detections in unprecedented conditions all along handling the potential lack of data
MakATT is mainly constituted of the following- 
Sample Collection pipeline: To collect the image samples of stakeholders, which acts as the training data for the model
Parition-1: Dividing the images and labels 
Label generation- The detections, coordinates from the image samples condensed into a json file format- which would be clubbed with the training data 
Augmentation- diversifying the data with the application of 8 parameters. The labels also being encrypted in the process
Partition pipeline- Partitioning the data for training, testing and validation purposes.
Detection and registration- the model shall then be used to perform live registration of stakeholders and data recording; mainly into Excel files

<img></img>

<h3>Sample collection pipeline-</h3>
 MarkATT has its own inbuilt capacity to take in the required amount of samples for any amount of people- and simultaneously saving it and integrating it to the Neural Network.
The collection pipeline first starts with establishing all the paths needed to ensure easy management of data, in further processing-


If the paths already happen to be established, it simply moves on! (no redundancy)

	What follows next is the segment to actually take in samples. The system is designed to
	Take in 30 images of each stakeholder, automatically in record 15 seconds and saving it 	
	To the paths established accordingly 
	
	This registration, can again be triggered by the administrator.
	What MarkATT features is customizing the memory allocation it uses based on the 
	Hardware capabilities of the system. This includes limiting the memory growth in the 	
	GPU (if it exists) used to train the model. Otherwise, it will train through the normal 
	Processor present in the administrator system.
	
	This ensures the prevention of system breakdowns when deployed at a large scale.
	
	
