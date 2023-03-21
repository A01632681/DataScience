## ----------------- ML TEST ----------------- ##
# Used to undestarnd the YOLO algorithm and how to use  the features it uses.
# The code is based on the tutorial from https://www.learnopencv.com/deep-learning-based-object-detection-using-yolov3-with-opencv-python-c/
# Also modified with other own ideas and codes.

## Way to call it: python Test.py -c yolov3.cfg -w yolov3.weights -cl yolov.txt

import cv2
import argparse
import numpy as np

# Argument parser to get the path to the image and the model.
ap = argparse.ArgumentParser()
#ap.add_argument('-i', '--image',
#                help = 'path to input image')
ap.add_argument('-c', '--config',
                help = 'path to yolo config file')
ap.add_argument('-w', '--weights',
                help = 'path to yolo pre-trained weights')
ap.add_argument('-cl', '--classes',
                help = 'path to text file containing class names')
args = ap.parse_args()

# Function to get the output layer names in the YOLO architecture.
def get_output_layers(net):
    
    layer_names = net.getLayerNames()
    try:
        output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
    except:
        output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

    return output_layers

# Function to draw bounding box on the detected object with class name.
def draw_prediction(img, class_id, confidence, x, y, x_plus_w, y_plus_h):

    label = str(classes[class_id])

    color = COLORS[class_id]

    cv2.rectangle(img, (x,y), (x_plus_w,y_plus_h), color, 2)

    cv2.putText(img, label +' '+str(round(confidence*100,2)), (x-10,y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

# Initialize the video capture object to get the real-time video feed from the webcam. / Change the 0 to the path to the video file to use a prerecorded video.
cap = cv2.VideoCapture (0)

# Set the scale factor for the image and initialize the classes variable
scale = 0.00392
classes = None

# Read the classes from the file and set a random color for each class.
with open(args.classes, 'r') as f:
    classes = [line.strip() for line in f.readlines()]
COLORS = np.random.uniform(0, 255, size=(len(classes), 3))

# Read the pre-trained model and config file.
net = cv2.dnn.readNet(args.weights, args.config)

# Read the video frame by frame and process it.
while cap.isOpened ():
    _, frame = cap.read ()
    if frame is None: break
    # image = cv2.imread(args.image)
    image = frame
    Width = image.shape[1]
    Height = image.shape[0]

    # Convert the image into a blob that can be input to the YOLO model
    blob = cv2.dnn.blobFromImage(image, scale, (416,416), (0,0,0), True, crop=False)
    
    # Set the input to the model
    net.setInput(blob)
    
    # Get the output layers from the YOLO model
    outs = net.forward(get_output_layers(net))

    # Initialize the lists for the class ids, confidences and the boxes.
    class_ids = []
    confidences = []
    boxes = []
    conf_threshold = 0.5
    nms_threshold = 0.4

    # Loop through each output layer from the YOLO model
    for out in outs:
        for detection in out:
            # Get the class id, confidence and the bounding box for each detection.
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > conf_threshold:
                center_x = int(detection[0] * Width)
                center_y = int(detection[1] * Height)
                w = int(detection[2] * Width)
                h = int(detection[3] * Height)
                x = center_x - w / 2
                y = center_y - h / 2
                class_ids.append(class_id)
                confidences.append(float(confidence))
                boxes.append([x, y, w, h])

    # This line uses Non-Maximum Suppression (NMS) algorithm to remove the overlapping bounding boxes and keep only the most confident ones for each object detected.
    indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)

    # Loop through the indices of the remaining boxes after NMS
    for i in indices:
        try:
            box = boxes[i]
        except:
            i = i[0]
            box = boxes[i]

        # Extract the x, y, w, and h coordinates from the box
        x = box[0]
        y = box[1]
        w = box[2]
        h = box[3]
        
        # Display the image with bounding boxes and labels
        draw_prediction(image, class_ids[i], confidences[i], round(x), round(y), round(x+w), round(y+h))
    
    # Let the user see what we have done.
    cv2.imshow('title', frame)
    
    # If Q is pressed, exit the loop.
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Tidy up OpenCV.
cap.release()
cv2.imwrite("object-detection.jpg", image)
cv2.destroyAllWindows()    