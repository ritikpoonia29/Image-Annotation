import cv2
import os
import xml.etree.ElementTree as ET

def create_xml_annotation(image_name, contours, output_folder, image_shape):
    # Create the XML annotation structure
    annotation = ET.Element("annotation")
    
    ET.SubElement(annotation, "filename").text = image_name
    size = ET.SubElement(annotation, "size")
    ET.SubElement(size, "width").text = str(image_shape[1])  # Image width
    ET.SubElement(size, "height").text = str(image_shape[0])  # Image height
    ET.SubElement(size, "depth").text = str(image_shape[2])  # Number of channels (RGB)
    
    for contour in contours:
        # Create object element for each contour (line segment)
        obj = ET.SubElement(annotation, "object")
        ET.SubElement(obj, "name").text = "line_segment"
        ET.SubElement(obj, "pose").text = "Unspecified"
        ET.SubElement(obj, "truncated").text = "0"
        ET.SubElement(obj, "difficult").text = "0"
        
        # Create bounding box coordinates (box around the contour)
        x, y, w, h = cv2.boundingRect(contour)
        bndbox = ET.SubElement(obj, "bndbox")
        ET.SubElement(bndbox, "xmin").text = str(x)
        ET.SubElement(bndbox, "ymin").text = str(y)
        ET.SubElement(bndbox, "xmax").text = str(x + w)
        ET.SubElement(bndbox, "ymax").text = str(y + h)
    
    # Write the XML annotation to a file
    xml_path = os.path.join(output_folder, image_name.replace(".jpg", ".xml"))
    tree = ET.ElementTree(annotation)
    tree.write(xml_path)

def annotate_images(input_folder, output_folder):
    print(f"Input folder: {input_folder}")
    print(f"Output folder: {output_folder}")
    
    # Check if the input folder exists
    if not os.path.exists(input_folder):
        print(f"Input folder does not exist: {input_folder}")
        return
    
    # Check if the output folder exists, if not create it
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # List all files in the input folder
    all_files = os.listdir(input_folder)
    
    print(f"Number of total files in folder: {len(all_files)}")  # Debugging print
    print(f"List of all files: {all_files}")  # Debugging print
    
    # Get the list of all image files in the input folder
    image_files = [f for f in all_files if f.lower().endswith('.jpg')]
    
    print(f"Number of JPG files found: {len(image_files)}")  # Debugging print
    print(f"List of JPG files: {image_files}")  # Debugging print
    
    for image_file in image_files:
        # Construct the full input path
        input_path = os.path.join(input_folder, image_file)
        
        print(f"Processing image: {input_path}")  # Debugging print
        
        # Read the image
        image = cv2.imread(input_path)
        
        if image is None:
            print(f"Error reading image: {input_path}")
            continue
        
        print(f"Image shape: {image.shape}")  # Debugging print
        
        # Convert image to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Detect edges using Canny edge detection
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        
        # Find contours
        contours = None
        hierarchy = None
        
        if cv2.__version__.startswith('3'):
            _, contours, hierarchy = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        else:
            contours, hierarchy = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        print(f"Number of contours found: {len(contours)}")  # Debugging print
        
        # Create XML annotation
        create_xml_annotation(image_file, contours, output_folder, image.shape)
        
        print(f'Annotated and saved {image_file}')



# Input and output folders
input_folder = "C:/Users/MY HP/Downloads/labelImg-master/Shapes"
output_folder = "C:/Users/MY HP/Downloads/labelImg-master/Annonated_XML"

# Call the function
annotate_images(input_folder, output_folder)
