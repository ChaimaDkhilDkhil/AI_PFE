import numpy as np
import cv2
import os
import tkinter
from tkinter import *
import pyautogui
from PIL import Image
import argparse
from datetime import datetime
import pyodbc
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image

# connexion  to the database
server = 'LAPTOP-NO435VT2\MSSQLSERVER01'
database = 'DefaultDatabase'
cnxn = pyodbc.connect(f'DRIVER={{SQL Server}};SERVER={server};DATABASE={database}')
cursor = cnxn.cursor()

# Parse command line arguments
ap = argparse.ArgumentParser()
ap.add_argument("-y", "--yolo", required=False, default="yolo_model",
                help="base path to YOLO directory")
ap.add_argument("-c", "--confidence", type=float, default=0.1,
                help="minimum probability to filter weak detections")
ap.add_argument("-t", "--threshold", type=float, default=0.1,
                help="threshold when applying non-maxima suppression")
args = vars(ap.parse_args())

# Read YOLO model and configuration
weightsPath = os.path.sep.join([args["yolo"], "custom.weights"])
configPath = os.path.sep.join([args["yolo"], "custom.cfg"])
net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)
ln = net.getLayerNames()
ln = [ln[i - 1] for i in net.getUnconnectedOutLayers()]

def insert_to_database(width_cm, height_cm, anomaly_image_data, full_screen_image_data, object_name):
    current_date = datetime.now()
    sql = "INSERT INTO anomalie (ObjectName, height, width, Date, Classification, laise, PointsAwarded) VALUES (?, ?, ?, ?, ?, ?, ?)"
    val = (object_name, height_cm, width_cm, current_date, None, None, None)

    try:
        cursor.execute(sql, val)
        cnxn.commit()

        row_count = cursor.rowcount
        if row_count == 0:
            print("Error: Values not inserted into anomalie")
        else:
            print("Values inserted successfully into anomalie")

    except pyodbc.Error as e:
        print(f"Error: {e}")
        print(f"Anomaly image size: {len(anomaly_image_data)} bytes")
        print(f"Full screen image size: {len(full_screen_image_data)} bytes")

def browse():
    # Open a file dialog to select an image file
    path = tkinter.filedialog.askopenfilename()

    # Check if a valid file path is selected
    if len(path) > 0:
        print(path)    
        # Read the selected image using OpenCV
        image = cv2.imread(path)
        (H, W) = image.shape[:2]

        # Prepare the image for object detection using YOLO
        blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416), swapRB=True, crop=False)
        net.setInput(blob)
        layerOutputs = net.forward(ln)

        # Initialize lists to store detection results
        boxes = []
        confidences = []
        classIDs = []

        # Loop over the output layers to process detections
        for output in layerOutputs:
            for detection in output:
                scores = detection[5:]
                classID = np.argmax(scores)
                confidence = scores[classID]

                # Check if the detected object's confidence is above a threshold
                if confidence > args["confidence"]:
                    # Extract bounding box coordinates and dimensions
                    box = detection[0:4] * np.array([W, H, W, H])
                    (centerX, centerY, width, height) = box.astype("int")

                    x = int(centerX - (width / 2))
                    y = int(centerY - (height / 2))

                    # Append the detection results to the lists
                    boxes.append([x, y, int(width), int(height)])
                    confidences.append(float(confidence))
                    classIDs.append(classID)

        # Apply non-maximum suppression to remove redundant bounding boxes
        idxs = cv2.dnn.NMSBoxes(boxes, confidences, args["confidence"], args["threshold"])

        # Check if any boxes remain after non-maximum suppression
        if len(idxs) > 0:
            for i in idxs.flatten():
                # Extract coordinates and dimensions of the remaining boxes
                (x, y) = (boxes[i][0], boxes[i][1])
                (w, h) = (boxes[i][2], boxes[i][3])
                
                # Draw a rectangle around the detected object
                color = (0, 255, 255)
                cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)

                # Calculate and display width and height in centimeters
                zone_size_y = 10
                zone_size_x = 10
                height_cm = h / image.shape[0] * zone_size_y
                width_cm = w / image.shape[1] * zone_size_x

                text = f"W: {width_cm:.2f} cm, H: {height_cm:.2f} cm"
                cv2.putText(image, text, (x, y + h + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

                # Save anomaly region screenshot and full-screen screenshot
                save_folder = r"C:\PFE\Fabric-Stain-Detection-finale\dist\main\images"
                anomaly_screenshot = pyautogui.screenshot(region=(int(x), int(y), int(w), int(h)))
                image_path = os.path.join(save_folder, f"anomaly_{len(os.listdir(save_folder))}.png")
                print(f"Saving anomaly screenshot to: {image_path}")
                anomaly_screenshot.save(image_path, quality=95)

                # Read and store image data for database insertion
                with open(image_path, "rb") as file:
                    image_data = file.read()

                # Capture unique full-screen screenshot for each object
                full_screen_screenshot = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
                full_screen_image_path = os.path.join(save_folder, f"full_screen_{len(os.listdir(save_folder))}.png")
                full_screen_screenshot.save(full_screen_image_path, quality=95)

                # Read and store full-screen image data for database insertion
                with open(full_screen_image_path, "rb") as file:
                    full_screen_image_data = file.read()

                # Insert detection details into the database
                insert_to_database(width_cm, height_cm, image_data, full_screen_image_data, f"Object_{len(os.listdir(save_folder))}")

        # Resize the image for display and show it
        image = cv2.resize(image, (640, 480))
        cv2.imshow("Image", image)
        cv2.waitKey(0)
        cv2.moveWindow("Image", 0, 0)


def train_and_predict_cnn():
    # Data augmentation for the training set to improve model generalization
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True
    )
    
    # Load and preprocess the training set using the specified data augmentation
    train_set = train_datagen.flow_from_directory(
        'Dataset',            # Directory containing the training images
        target_size=(64, 64),  # Resize images to 64x64 pixels
        batch_size=32,         # Batch size for training
        class_mode='sparse'    # Use sparse categorical labels
    )

    # Preprocess the test set without data augmentation
    test_datagen = ImageDataGenerator(rescale=1./255)
    test_set = test_datagen.flow_from_directory(
        'test_set',           # Directory containing the test images
        target_size=(64, 64),  # Resize images to 64x64 pixels
        batch_size=32,         # Batch size for testing
        class_mode='sparse'    # Use sparse categorical labels
    )

    # Create a Sequential model for the CNN
    cnn = tf.keras.models.Sequential()
    
    # Add the first convolutional layer with ReLU activation and input shape
    cnn.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu', input_shape=[64, 64, 3]))

    # Add max pooling layer to reduce spatial dimensions
    cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))

    # Add a second convolutional layer with ReLU activation
    cnn.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu'))

    # Add another max pooling layer
    cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))

    # Flatten the output before passing it to fully connected layers
    cnn.add(tf.keras.layers.Flatten())

    # Add a fully connected layer with 128 units and ReLU activation
    cnn.add(tf.keras.layers.Dense(units=128, activation='relu'))

    # Output layer with 2 units for binary classification and softmax activation
    cnn.add(tf.keras.layers.Dense(units=2, activation='softmax'))

    # Compile the model with Adam optimizer, sparse categorical crossentropy loss, and accuracy metric
    cnn.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # Train the model on the training set and validate on the test set for 25 epochs
    cnn.fit(x=train_set, validation_data=test_set, epochs=25)
    
    # Return the trained CNN model
    return cnn


def predict_cnn_result(cnn, test_image):
    # Resize the input test image to match the model's input size (64x64 pixels)
    test_image = cv2.resize(test_image, (64, 64))  

    # Normalize the pixel values to the range [0, 1] and add a dimension for training
    test_image = np.expand_dims(test_image, axis=0) / 255.0  

    # Use the trained CNN model to predict the class of the input image
    result = cnn.predict(test_image)

    # Check the index of the highest predicted probability and assign the corresponding class
    if np.argmax(result[0]) == 0:
        prediction = 'HOLE'
    else:
        prediction = 'LEISURE'

    # Return the predicted class
    return prediction


def calculate_points(width_cm, height_cm):
    points = 0
    
    if width_cm >= 0 and width_cm <= 7.5:
        points = 1
    elif width_cm > 7.5 and width_cm <= 15:
        points = 2
    elif width_cm > 15 and width_cm <= 23:
        points = 3
    elif width_cm > 23 and width_cm <= 100:  
        points = 4
    
    if height_cm >= 0 and height_cm <= 7.5:
        points = 1
    elif height_cm > 7.5 and height_cm <= 15:
        points = 2
    elif height_cm > 15 and height_cm <= 23:
        points = 3
    elif height_cm > 23 and height_cm <= 100:  
        points = 4
        
    return points

def preprocess_image(image):
    # Convert the input image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply Sobel operator to compute the gradient in the x and y directions
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    
    # Combine the x and y gradients to get the overall gradient magnitude
    sobel = np.sqrt(sobelx**2 + sobely**2)
    
    # Normalize the gradient magnitude to the range [0, 255] for better visualization
    sobel = cv2.normalize(sobel, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
    
    # Return the preprocessed image with gradient information
    return sobel

def capture_and_detect():
    # Open a connection to the default camera (index 0)
    
    cap = cv2.VideoCapture(0)
    anomaly_count = 1
    
    # Train the CNN model only once before entering the loop
    cnn_model = train_and_predict_cnn()  

    while True:
        # Read a frame from the camera
        ret, frame = cap.read()
        if not ret:
            break

        # Check and convert the frame to RGB format if needed
        if len(frame.shape) == 2:
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
        elif frame.shape[2] == 1:
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

        # Prepare the frame for object detection using YOLO
        blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416), swapRB=True, crop=False)
        net.setInput(blob)
        layerOutputs = net.forward(ln)

        # Loop over the output layers to process detections
        for output in layerOutputs:
            for detection in output:
                scores = detection[5:]
                classID = np.argmax(scores)
                confidence = scores[classID]

                # Check if the detected object's confidence is above a threshold
                if confidence > args["confidence"]:
                    # Extract bounding box coordinates and dimensions
                    box = detection[0:4] * np.array([frame.shape[1], frame.shape[0], frame.shape[1], frame.shape[0]])
                    (centerX, centerY, width, height) = box.astype("int")
                    x = int(centerX - (width / 2))
                    y = int(centerY - (height / 2))

                    # Ensure bounding box coordinates are within image boundaries
                    x = max(0, x)
                    y = max(0, y)
                    width = min(frame.shape[1] - x, width)
                    height = min(frame.shape[0] - y, height)

                    # Draw a rectangle around the detected object
                    cv2.rectangle(frame, (x, y), (x + width, y + height), (0, 255, 255), 2)
                    
                    # Calculate and display width and height in centimeters
                    zone_size_x = 10
                    zone_size_y = 10
                    width_cm = width / frame.shape[1] * zone_size_x
                    height_cm = height / frame.shape[0] * zone_size_y
                    text = f"W: {width_cm:.2f} cm, H: {height_cm:.2f} cm"
                    cv2.putText(frame, text, (x, y - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
                    
                    # Calculate and display points based on width and height
                    points = calculate_points(width_cm, height_cm)
                    cv2.putText(frame, f"Points: {points}", (x, y - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
                    import os

                    

                    
                    # Save the full screen and anomaly region screenshots
                    save_folder = r"C:\PFE\Fabric-Stain-Detection-finale\dist\main\images"
                    full_screen_screenshot = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                    full_screen_image_path = os.path.join(save_folder, f"full_screen_{anomaly_count}.png")
                    full_screen_screenshot.save(full_screen_image_path)

                    anomaly_region = frame[y:y + height, x:x + width]
                    anomaly_screenshot = Image.fromarray(cv2.cvtColor(anomaly_region, cv2.COLOR_BGR2RGB))
                    anomaly_screenshot_path = os.path.join(save_folder, f"anomaly_{anomaly_count}.png")
                    anomaly_screenshot.save(anomaly_screenshot_path)

                    # Make a prediction using the CNN model on the anomaly region
                    prediction = predict_cnn_result(cnn_model, np.array(anomaly_screenshot))
                    cv2.putText(frame, f"Prediction: {prediction}", (x, y - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

                    # Increment the anomaly count
                    anomaly_count += 1

        # Display the frame with object detection results
        cv2.imshow("Camera", frame)

        # Break the loop if 'q' key is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the camera and close the window
    cap.release()
    cv2.destroyWindow("Camera")


# Create a GUI window using Tkinter
root = Tk()
root.geometry("600x300")

# Add a label to the GUI window
label = Label(root, text="WIC MIC fabric default detection using IA", font=("Courier", 14, "italic", "bold"))
label.place(x=310, y=40, anchor="center")

# Confidence values for YOLO object detection
confidence_values = [0.8, 0.5, 0.4, 0.3, 0.2, 0.1]
confidence_var = StringVar(root)
confidence_var.set(confidence_values[5])

# Function to update the confidence value based on user selection
def update_confidence(value):
    args["confidence"] = value

# Create a dropdown menu for selecting confidence value
dropdown_label = Label(root, text="Choose confidence")
dropdown_label.place(x=100, y=150, anchor="center")
dropdown = OptionMenu(root, confidence_var, *confidence_values, command=update_confidence)
dropdown.place(x=200, y=150, anchor="center")

# Button to browse for images (not defined in the provided code)
btn = Button(root, text="Browse", command=browse)
btn.place(x=300, y=150, anchor="center")

# Button to start camera and object detection
btn_camera = Button(root, text="Camera", command=capture_and_detect)
btn_camera.place(x=400, y=150, anchor="center")

# Start the Tkinter event loop
root.mainloop()

