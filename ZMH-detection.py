import jetson.inference
import jetson.utils


# --------------------------------------------------
# Load SSD-Mobilenet-v2 object detection model
# --------------------------------------------------
net = jetson.inference.detectNet(
    "ssd-mobilenet-v2",
    threshold=0.5
)

# Font used to draw extra detection information
font = jetson.utils.cudaFont()


# --------------------------------------------------
# Input and output images
# --------------------------------------------------
images = [
    (
        "/home/nvidia/Desktop/Image1.jpg",
        "/home/nvidia/Desktop/result1.jpg"
    ),
    (
        "/home/nvidia/Desktop/Image2.jpg",
        "/home/nvidia/Desktop/result2.jpg"
    )
]


# --------------------------------------------------
# Process the two images
# --------------------------------------------------
for i, (input_path, output_path) in enumerate(images, start=1):

    print("\n======================================")
    print("IMAGE", i)
    print("======================================")

    # Load image
    img = jetson.utils.loadImage(input_path)

    # Run object detection
    # box     = bounding box
    # labels  = class name
    # conf    = confidence
    detections = net.Detect(
        img,
        overlay="box,labels,conf"
    )

    # If nothing was detected
    if len(detections) == 0:
        print("No object detected.")
        continue

    # --------------------------------------------------
    # Select the detection with the highest confidence
    # --------------------------------------------------
    detection = max(
        detections,
        key=lambda x: x.Confidence
    )

    class_name = net.GetClassDesc(detection.ClassID)

    # --------------------------------------------------
    # Information shown on the detected image
    # Similar to the teacher's example
    # --------------------------------------------------
    info = (
        f"L: {detection.Left:.0f}  "
        f"T: {detection.Top:.0f}  "
        f"W: {detection.Width:.0f}  "
        f"H: {detection.Height:.0f}  "
        f"Area: {detection.Area:.0f}"
    )

    # Position the extra information below the class label
    text_x = int(detection.Left)
    text_y = int(detection.Top + 25)

    # Draw information onto image
    font.OverlayText(
        img,
        img.width,
        img.height,
        info,
        text_x,
        text_y,
        font.White,
        font.Gray40
    )

    # --------------------------------------------------
    # Print all Assignment 3 required information
    # --------------------------------------------------
    print("ClassID:", detection.ClassID)
    print("Class Name:", class_name)
    print("Confidence:", detection.Confidence)

    print("Left:", detection.Left)
    print("Top:", detection.Top)
    print("Right:", detection.Right)
    print("Bottom:", detection.Bottom)

    print("Width:", detection.Width)
    print("Height:", detection.Height)

    print("Area:", detection.Area)
    print("Center:", detection.Center)

    # --------------------------------------------------
    # Save detected image
    # --------------------------------------------------
    jetson.utils.saveImage(
        output_path,
        img
    )

    print("Detected image saved to:", output_path)