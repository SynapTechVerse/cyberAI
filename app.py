import os
import subprocess
import time
import gc
import logging

import cv2
import face_recognition
import numpy as np
from PIL import Image, ImageFile
# pip install psutil
import psutil


# Set up logging configuration
logging.basicConfig(level=logging.INFO, format='%(message)s', handlers=[logging.StreamHandler()])
logger = logging.getLogger()

ImageFile.LOAD_TRUNCATED_IMAGES = True

# Start time
start_time = time.time()
counter = 0
BATCH_SIZE = 10  # Number of images to process at a time

def repair_jpeg_image_with_jpegtran(image_path):
    """Repair JPEG files using jpegtran tool."""
    repaired_image_path = image_path + "_repaired.jpg"
    try:
        subprocess.run(["jpegtran", "-copy", "none", "-optimize", "-outfile", repaired_image_path, image_path], check=True)
        logger.info(f"Repaired JPEG image using jpegtran: {image_path}")
        return repaired_image_path
    except Exception as e:
        logger.error(f"Error repairing JPEG image with jpegtran {image_path}: {e}")
        return None

def repair_jpeg_image_with_pillow(image_path):
    """Repair JPEG files using Pillow by re-saving them."""
    try:
        img = Image.open(image_path)
        img.save(image_path, 'JPEG', quality=95)  # Re-save with a fresh encoding
        logger.info(f"Repaired JPEG image using Pillow: {image_path}")
        return image_path
    except Exception as e:
        logger.error(f"Error repairing JPEG image {image_path}: {e}")
        return None

def fix_png_image(image_path):
    """Fix PNG files with corrupt color profiles by removing the ICC profile."""
    try:
        img = Image.open(image_path)
        img = img.convert('RGB')  # This automatically removes the ICC profile by converting to RGB
        img.save(image_path, 'PNG', quality=95)  # Save the image back without the ICC profile
        logger.info(f"Fixed PNG image: {image_path}")
        return True
    except Exception as e:
        logger.error(f"Error while fixing PNG image {image_path}: {e}")
        return False

def create_directory_if_not_exists(directory_path):
    """Ensure the directory exists, if not, create it."""
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
        logger.info(f"Created directory: {directory_path}")

def is_image_file(filename):
    """Check if a file is a valid image based on file extension."""
    valid_extensions = ('.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff')
    return filename.lower().endswith(valid_extensions)

def load_image(folder, filename):
    global counter
    
    """Load a single image and repair if necessary."""
    if not is_image_file(filename):
        logger.debug(f"Skipping non-image file: {filename}")
        return None, False
    file_path = os.path.join(folder, filename)
    img = cv2.imread(file_path)
    if img is None:
        logger.warning(f"OpenCV failed to read {filename}, trying with Pillow.")
        
        # Handle PNG images with corrupt profiles and CRC errors
        if filename.lower().endswith('.png'):
            if fix_png_image(file_path):  # Attempt to fix with Pillow
                img = cv2.imread(file_path)  # Try to load again after fixing
        # Handle JPEG images with possible corruption
        elif filename.lower().endswith('.jpg') or filename.lower().endswith('.jpeg'):
            repaired_path = repair_jpeg_image_with_jpegtran(file_path)
            if repaired_path is None:
                repaired_path = repair_jpeg_image_with_pillow(file_path)
            if repaired_path:
                img = cv2.imread(repaired_path)  # Try to load repaired JPEG
    
    if img is not None:
        logger.debug(f"{counter}...Image Loaded: {filename}")
        counter += 1
        return img, True
    else:
        logger.error(f"Could not load or repair image {filename}. Skipping.")
        return filename, False

def get_face_encodings(images):
    """Get face encodings for a list of images."""
    encodings = []
    for img in images:
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        faces = face_recognition.face_locations(img_rgb)
        if faces:
            face_encodings = face_recognition.face_encodings(img_rgb, faces)
            encodings.extend(face_encodings)
        else:
            logger.debug("No faces found in image.")
    logger.debug(f"Total face encodings: {len(encodings)}")
    return encodings

def match_face(img, folder_a_encodings, threshold=0.5):
    """Match a single image's face encoding to Folder A's encodings."""
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    faces_b = face_recognition.face_locations(img_rgb)
    if faces_b:
        face_b_encodings = face_recognition.face_encodings(img_rgb, faces_b)
        for face_b_encoding in face_b_encodings:
            # Compare faces with a custom threshold
            distances = face_recognition.face_distance(folder_a_encodings, face_b_encoding)
            best_match_index = np.argmin(distances)  # Find the index of the closest match
            if distances[best_match_index] < threshold:  # Use the threshold to avoid false positives
                logger.debug(f"Match found for {img}. Distance: {distances[best_match_index]}")
                return True
    return False

def match_faces_in_batch(folder_b_images, batch_filenames,folder_a_encodings, threshold=0.5):
    """Match faces in Folder B's batch of images against Folder A's encodings."""
    #matched_images = []
    matche_img_dict={}
    for img,img_name in zip(folder_b_images,batch_filenames):
        if match_face(img, folder_a_encodings, threshold):
            #matched_images.append(img_name)
            matche_img_dict[img_name]=img
        # Explicitly release the image after processing to free up memory
    #logger.debug(f"{len(matched_images)} matched images in this batch.")
    gc.collect()
    return matche_img_dict
    #return matched_images

def load_images_from_folder(folder, batch_size=BATCH_SIZE):
    """Load images in batches from the folder."""
    images = []
    filenames = []
    
    # Initialize batch variables here before starting to process the folder
    batch_images = []
    batch_filenames = []
    
    # Ensure we are checking files in the folder
    files = os.listdir(folder)

    for idx, filename in enumerate(files):
        img, is_img_file = load_image(folder, filename)
        if is_img_file:
            batch_images.append(img)
            batch_filenames.append(filename)
        
        # Process the batch when it's full or if it's the last file
        #if psutil.virtual_memory().percent > 90:
        #    logger.warning("Memory usage exceeds 90%. Stopping loading images.")
        #    break

        #if psutil.virtual_memory().percent > 90 or idx == len(files) - 1:
        
        if len(batch_images) >= batch_size or idx == len(files) - 1:
            logger.warning("Memory usage exceeds 90%. Processing till this batch.")
            images.append(batch_images)
            filenames.append(batch_filenames)
            batch_images = []  # Reset the batch
            batch_filenames = []  # Reset the batch
            # Clear memory and reset for the next batch
            gc.collect()
    
    return zip(images, filenames)

def save_images2(image_dict,output_folder):
    #Save images to the specified output folder.
    create_directory_if_not_exists(output_folder)
    for img_name,img in image_dict.items():
        output_path = os.path.join(output_folder, img_name)
        cv2.imwrite(output_path, img)
        logger.info(f"Saved matched image to {output_path}")


def write_list_to_file(my_list, file_name):
    # Create the 'check_point' folder if it doesn't exist
    folder_name = 'check_point'
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
    
    # Construct the full path to the file
    file_path = os.path.join(folder_name, file_name)
    
    # Open the file in write mode and write the list to it
    with open(file_path, 'w') as file:
        for item in my_list:
            file.write(f"{item}\n")

if __name__ == '__main__':
    # Define the main function
    def main():
        folder_a_path = "/app/img/document/a"  # Path to Folder A
        folder_b_path = "/app/img/document/b"  # Path to Folder B
        output_folder = "/app/output"          # Path to Output Folder 

        # Ensure the directories exist
        create_directory_if_not_exists(folder_a_path)
        create_directory_if_not_exists(folder_b_path)

        logger.info("Starting to load images from Folder A...")
        # Load images from Folder A in batches
        folder_a_images, folder_a_filenames = [], []
        total_images_loaded = 0  # Initialize counter for total images loaded
        for batch_images, batch_filenames in load_images_from_folder(folder_a_path):
            total_images_loaded += len(batch_images)  # Increment counter by the number of images in the batch
            folder_a_images.extend(batch_images)
            folder_a_filenames.extend(batch_filenames)
            del batch_images, batch_filenames  # Explicitly delete batch variables
        logger.info(f"Criminal Face (A) Data Loaded. Total images loaded: {total_images_loaded}")

        logger.info("Starting to get face encodings for Folder A...")
        # Get face encodings for images in Folder A in batches
        folder_a_encodings = get_face_encodings(folder_a_images)

        # Clear memory for folder_a_images after processing
        del folder_a_images
        gc.collect()
        logger.info("Criminal Face Encoding Completed and memory released for folder A.")

        logger.info("Starting to process Folder B...")
        total_images_number = len([f for f in os.listdir(folder_b_path) if os.path.isfile(os.path.join(folder_b_path, f))])
        total_batches=total_images_number//BATCH_SIZE
        logger.info(f"total number of files to check for criminal(B) :{total_images_number} , so total batches would be {total_batches}")
        # Process Folder B in batches
        matched_images = []
        for batch in range(total_batches):
            for batch_images, batch_filenames in load_images_from_folder(folder_b_path):
                logger.info(f"Processing batch    Batch {batch+1} / {total_batches}    which has {len(batch_images)} images from Folder B...")
                # Match faces in Folder B with Folder A encodings
                #matched_batch = match_faces_in_batch(batch_images, batch_filenames, folder_a_encodings, threshold=0.5)
                #write_list_to_file(matched_batch, "batch_"+str(batch+1))
                
                matched_batch_dict = match_faces_in_batch(batch_images, batch_filenames, folder_a_encodings, threshold=0.5)
                
                save_images2(matched_batch_dict,output_folder)
                logger.info(f"Batch {batch+1} processed successfully.")
                del matched_batch_dict
                #del matched_batch
                gc.collect()
                print("Thank you!")
                break

        #logger.info(f"Matched {len(matched_images)} images.")'
        
        # End time
        end_time = time.time()
        execution_time = end_time - start_time
        logger.info(f"Done. Execution time: {execution_time} seconds")

    main()







"""
def save_images(images, output_folder):
    #Save images to the specified output folder.
    create_directory_if_not_exists(output_folder)
    for idx, img in enumerate(images):
        output_path = os.path.join(output_folder, f"matched_{idx}.jpg")
        cv2.imwrite(output_path, img)
        logger.info(f"Saved matched image to {output_path}")
    folder_a_path = "/app/img/document/a"  # Path to Folder A
    folder_b_path = "/app/img/document/b"  # Path to Folder B

    # Ensure the directories exist
    create_directory_if_not_exists(folder_a_path)
    create_directory_if_not_exists(folder_b_path)
    
    logger.info("Starting to load images from Folder A...")
    # Load images from Folder A in batches
    folder_a_images, folder_a_filenames = [], []
    total_images_loaded = 0  # Initialize counter for total images loaded
    for batch_images, batch_filenames in load_images_from_folder(folder_a_path):
        total_images_loaded += len(batch_images)  # Increment counter by the number of images in the batch
        folder_a_images.extend(batch_images)
        folder_a_filenames.extend(batch_filenames)
        del batch_images, batch_filenames  # Explicitly delete batch variables
    logger.info(f"Criminal Face (A) Data Loaded. Total images loaded: {total_images_loaded}")
    
    logger.info("Starting to get face encodings for Folder A...")
    # Get face encodings for images in Folder A in batches
    folder_a_encodings = get_face_encodings(folder_a_images)
    
    # Clear memory for folder_a_images after processing
    del folder_a_images
    gc.collect()
    logger.info("Criminal Face Encoding Completed and memory released for folder A.")

    logger.info("Starting to process Folder B...")
    # Process Folder B in batches
    matched_images = []
    for batch_images, batch_filenames in load_images_from_folder(folder_b_path):
        logger.info(f"Processing batch of {len(batch_images)} images from Folder B...")
        # Match faces in Folder B with Folder A encodings
        #matched_batch = match_faces_in_batch(batch_images, batch_filenames, folder_a_encodings, threshold=0.5)
        matched_images.extend(matched_batch)
        # Clear memory after processing each batch
        gc.collect()

    logger.info(f"Matched {len(matched_images)} images.")

    # Save matched images to the output folder
    output_folder = "/app/output"
    #save_images(matched_images, output_folder)
    

    # End time
    end_time = time.time()
    execution_time = end_time - start_time
    logger.info(f"Done. Execution time: {execution_time} seconds")
"""
