#importing cv2, an the most widely used computer vision library, which contains tools that makes our life
#easier while dealing with image processing and object detection
import cv2


#global variables
mean = 127.5
frame_width = 320
frame_height = 320

#colors
blue = (255,0,0)
green = (0,255,0)
red = (0,0,255)
white = (255,255,255)

############################## LOADING AND READING MODEL FILES ##################################


#File containing the names of all the objects whose models are available with us
list_of_object_names = "D:/IIT/Academics/5fth Semester/ECN-343/Project/Files/ObjectNames.names"

#File in which network configuration of the object models are defined
configurations = "D:/IIT/Academics/5fth Semester/ECN-343/Project/Files/ModelsConfiguration.pbtxt"

#This is binary file which contains the weights of the pixels of the trained models
weights = "D:/IIT/Academics/5fth Semester/ECN-343/Project/Files/BinaryWeights.pb"

#dnn_DetectionModel creates dnn networkwork using weights and config files which is stored in the object network
network = cv2.dnn_DetectionModel(weights,configurations)

#loading the object into a list
with open(list_of_object_names,"rt") as list:
    all_objects = list.read().rstrip("\n").split("\n")

#function which returns the list of objects to be detected
def get_list(selective=0):

    if(selective):    return ["person"]
    else: 
        return all_objects


############################## DEFINING THE FRAME PROPERTIES ##################################


#setting up the mean value of the rgb channels
network.setInputMean((mean,mean,mean))

#Scalefactor by which the frame will be scaled before running the detect method
network.setInputScale(1/mean)

#Setting up the size of the frame
network.setInputSize(frame_width,frame_height)

#This method will swap the first and last channels to extract more data from the image
network.setInputSwapRB(True)


#################################### CAPTURING VIDEO FROM CAMERA ###################################


#capturing the video stream from the default camera
videostream = cv2.VideoCapture(0)

#setting up the stream height
videostream.set(cv2.CAP_PROP_FRAME_HEIGHT,480)

#setting up the stream width
videostream.set(cv2.CAP_PROP_FRAME_WIDTH,640)


####################### RUNNING DNN ON THE FRAME AND DISPLAYING THE RESULT #########################


#getting the list of the objects to be identified
list = get_list(0)

#function to generate infinite loop
def infloop():
    while 1: yield 1
for _ in infloop():

    #reading the video stream by taking snapshots at fixed intervals
    frame = videostream.read()
    (checker,image) = frame

    
    if list: pass
    else:
        cv2.imshow("Output",image)
        if cv2.waitKey(1) == ord('s'):
            break
        continue

    #detect function runs the network over the frame and returns the results of the detection
    results = network.detect(image, 0.5, 0.5)
    (position,prob,boundaries) = results

    if len(position)==0: 
        if cv2.waitKey(1) == ord('s'):
            break
        continue

    #flattening the multidimensional ndarrays to use them together in a loop
    idx = position.flatten()
    probabilities = prob.flatten()
    shapes = boundaries
    final_data = zip(idx,probabilities,shapes)

    for i in final_data:

        (j,p,r) = i
        name = all_objects[j-1]
        if not (name in list): continue
        confidence = str(round(100*p,3))+" %"
        rect = r

        #these colors are used to set the values of the linetype and font of the box
        font1 = 2
        font2 = 2
        scalefactor = 0.65
        linetype = 2

        #creating the box with desired values of font, color and borderline
        cv2.putText(image, name.upper(),  (rect[0]+10, rect[1]+30), font1, scalefactor, blue,  linetype)
        cv2.putText(image, confidence,  (rect[0]+10, rect[1]+60), font2, scalefactor, red,  linetype)
        cv2.rectangle(image,  rect,  green,  thickness=2)

    #displaying the the final version of the frame
    cv2.imshow("Output",image)
    if cv2.waitKey(1) == ord('s'):
        break
################################### END OF THE PROGRAM ####################################

#available fonts in cv2

#   FONT_HERSHEY_SIMPLEX = 0
#   FONT_HERSHEY_PLAIN = 1
#   FONT_HERSHEY_DUPLEX = 2
#   FONT_HERSHEY_COMPLEX = 3
#   FONT_HERSHEY_TRIPLEX = 4
#   FONT_HERSHEY_COMPLEX_SMALL = 5
#   FONT_HERSHEY_SCRIPT_SIMPLEX = 6
#   FONT_HERSHEY_SCRIPT_COMPLEX = 7


#available linetypes in cv2

#   FILLED = -1,
#   LINE_4 = 4,
#   LINE_8 = 8,
#   LINE_AA = 16