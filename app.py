import cv2
import os
import face_recognition
import numpy as np
from PIL import Image, ImageFile
import subprocess
import time
import gc

ImageFile.LOAD_TRUNCATED_IMAGES = True


def repair_jpeg_image_with_jpegtran(image_path):
    """Repair JPEG files using jpegtran tool."""
    repaired_image_path = image_path + "_repaired.jpg"
    try:
        subprocess.run(["jpegtran", "-copy", "none", "-optimize", "-outfile", repaired_image_path, image_path], check=True)
        print(f"Repaired JPEG image using jpegtran: {image_path}")
        return repaired_image_path
    except Exception as e:
        print(f"Error repairing JPEG image with jpegtran {image_path}: {e}")
        return None

def repair_jpeg_image_with_pillow(image_path):
    """Repair JPEG files using Pillow by re-saving them."""
    try:
        img = Image.open(image_path)
        img.save(image_path, 'JPEG', quality=95)  # Re-save with a fresh encoding
        print(f"Repaired JPEG image using Pillow: {image_path}")
        return image_path
    except Exception as e:
        print(f"Error repairing JPEG image {image_path}: {e}")
        return None

def fix_png_image(image_path):
    """Fix PNG files with corrupt color profiles by removing the ICC profile"""
    try:
        img = Image.open(image_path)
        img = img.convert('RGB')  # This automatically removes the ICC profile by converting to RGB
        img.save(image_path, 'PNG', quality=95)  # Save the image back without the ICC profile
        print(f"Fixed PNG image: {image_path}")
        return True
    except Exception as e:

        print(f"Error while fixing PNG image {image_path}: {e}")
        return False

def create_directory_if_not_exists(directory_path):
    """Ensure the directory exists, if not, create it."""
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)

def is_image_file(filename):
    """Check if a file is a valid image based on file extension."""
    valid_extensions = ('.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff')
    return filename.lower().endswith(valid_extensions)

def load_images_from_folder_a(folder):
    images = []
    filenames = []
    corrupted_files = 0
    total_files = 0
    
    for filename in os.listdir(folder):
        total_files += 1
        file_path = os.path.join(folder, filename)
        
        # Skip non-image files like metadata files (e.g., .DS_Store or ._ files)
        if not is_image_file(filename):
            print(f"Skipping non-image file: {filename}")
            continue
        
        # Try loading the image with OpenCV
        img = cv2.imread(file_path)
        if img is None:
            print(f".......Warning: OpenCV failed to read {filename}, trying with Pillow.")
            
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
        
        if img is not None:# this should be an image 
            print("...RAVI : Image is True _> ",filename)
            images.append(img) 
            filenames.append(filename)
        else:
            corrupted_files += 1
            print(f"Warning: Could not load or repair image {filename}. Skipping.")
    
    # Calculate the percentage of corrupted files
    if total_files > 0:
        corrupted_percentage = (corrupted_files / total_files) * 100
    else:
        corrupted_percentage = 0
    print("...... RAVI : Corrupted percentage ->",round(corrupted_percentage,2),"%")
    
    return images, filenames, corrupted_percentage

def load_images_from_folder_b(folder,folder_a_encodings):
    images = []
    filenames = []
    corrupted_files = 0
    total_files = 0
    final_count= 0
    matched_file_names_from_b=[]

    
    for filename in os.listdir(folder):
        total_files += 1
        final_count +=1
        print("Scanning file :",final_count)# 5k images in 30 min
        file_path = os.path.join(folder, filename)
        
        # Skip non-image files like metadata files (e.g., .DS_Store or ._ files)
        if not is_image_file(filename):
            #print(f"Skipping non-image file: {filename}")
            continue
        
        # Try loading the image with OpenCV
        img = cv2.imread(file_path)
        if img is None:
            #print(f".......Warning: OpenCV failed to read {filename}, trying with Pillow.")
            
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
        
        if img is not None:# this should be an image 
            #print("...RAVI : Image is True _> ",filename)
            images.append(img) 
            filenames.append(filename)
        else:
            corrupted_files += 1
            print(f"Warning: Could not load or repair image {filename}. Skipping.")
        
        if total_files>200:
            matches = match_faces(images, folder_a_encodings, filenames, threshold=0.5)
            print("Number of files processd till now",total_files,' \n Writing data to file...')
            result_file=write_data(matches,"matched_files_"+str(final_count))
            #matched_file_names_from_b.extend(matches)
            matches.clear()
            images.clear()
            filenames.clear()
            total_files=0
            print("Data is written to ",result_file," and Memory released")
            
            

        gc.collect()
    
    # Calculate the percentage of corrupted files
    if total_files > 0:
        corrupted_percentage = (corrupted_files / total_files) * 100
    else:
        corrupted_percentage = 0
    print("...... RAVI : Corrupted percentage ->",round(corrupted_percentage,2),"%")
    
    return matched_file_names_from_b

def get_face_encodings(images):
    encodings = []
    for img in images:
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        faces = face_recognition.face_locations(img_rgb)
        if faces:
            face_encodings = face_recognition.face_encodings(img_rgb, faces)
            encodings.extend(face_encodings)
    return encodings

def match_faces(folder_b_images, folder_a_encodings, folder_b_filenames, threshold=0.5):
    matches = []
    for img, filename in zip(folder_b_images, folder_b_filenames):
        print("Processing File : ",filename)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        faces_b = face_recognition.face_locations(img_rgb)
        if faces_b:
            face_b_encodings = face_recognition.face_encodings(img_rgb, faces_b)
            for face_b_encoding in face_b_encodings:
                # Compare faces with a custom threshold
                distances = face_recognition.face_distance(folder_a_encodings, face_b_encoding)
                best_match_index = np.argmin(distances)  # Find the index of the closest match
                if distances[best_match_index] < threshold:  # Use the threshold to avoid false positives
                    #matches.append((filename, distances[best_match_index]))
                    matches.append(filename)
    return matches

def write_data(matched_file_names_from_b, file_name):
    os.makedirs("/app", exist_ok=True)
    app_directory = "/app"
    os.chmod(app_directory, 0o777)
    with open("/app/output/"+file_name+".txt", "w") as file:
        # Write each item from the list to the file
        for item in matched_file_names_from_b:
            file.write(item + "\n")


def main():
    folder_a_path = "/app/img/document/a"  # Path to Folder A
    folder_b_path = "/app/img/document/b"  # Path to Folder B

    # Ensure the directories exist
    create_directory_if_not_exists(folder_a_path)
    create_directory_if_not_exists(folder_b_path)
    
    # Load images from Folder A and Folder B
    folder_a_images, folder_a_filenames, _ = load_images_from_folder_a(folder_a_path)
    print(".........Criminal Face (A) Data Loaded..................")
    folder_a_encodings = get_face_encodings(folder_a_images)
    del folder_a_images
    gc.collect()
    print("Criminal Face Encoding Completed and memory released for folder A.")

    matched_file_names_from_b= load_images_from_folder_b(folder_b_path,folder_a_encodings)
    #matched_file_names_from_b=['1','2']
    print("SUCESSSSSSSSSSSSSSS got matching data")
    # Open a text file in write mode
    # Ensure the directory exists
    
    # output_directory = '/app/output'

    # # Check if directory exists
    # if not os.path.exists(output_directory):
    #     os.makedirs(output_directory, mode=0o777)  # Create the directory if it doesn't exist

    # # File path
    # file_path = os.path.join(output_directory, "result.txt")

    # try:
    #     with open(file_path, "w") as file:
    #         # Write each item from the list to the file
    #         matched_file_names_from_b = ["file1.txt", "file2.txt", "file3.txt"]  # Example list
    #         for item in matched_file_names_from_b:
    #             file.write(item + "\n")
    #     print(f"File written successfully to {file_path}")
    # except Exception as e:
    #     print(f"Error writing to file: {e}")


    # print("File 'output2.txt' has been created and written to successfully.",os.getcwd())

    # Find matches in Folder B based on encodings from Folder A
    

    

    

if __name__ == '__main__':
    main()