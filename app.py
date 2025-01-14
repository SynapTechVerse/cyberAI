# docker run -it -v C:/Ravi/c_ai/output:/app/output criminal

import os
import shutil
from collections import Counter


def main():
    # Define the paths for the folders
    output_folder = 'output'
    folder_b = 'img/document/b'
    

    # Step 1: Read all text files in the 'output' folder and store file names into a list
    matched_files_list = []
    for filename in os.listdir(output_folder):
        if filename.endswith(".txt"):
            with open(os.path.join(output_folder, filename), 'r') as file:
                # Add the file content (assuming one file name per line)
                matched_files_list.extend(file.read().splitlines())
    
    
    # Step 2: Count duplicates in matched_files_list
    """
    file_counts = Counter(matched_files_list)
    
    dup_count=0
    for file, count in file_counts.items():
        if count > 1:  # Print only duplicates
            dup_count+=1
    print("Duplicates :",dup_count," Total Files",len(matched_files_list))
    """


    suspect_present_folder = 'suspect_persent'
    suspect_abscent_folder = 'suspect_abscent'
    # Step 2: Ensure that the suspect folders exist
    os.makedirs(suspect_present_folder, exist_ok=True)
    os.makedirs(suspect_abscent_folder, exist_ok=True)

    count_files_movement=0
    # Step 3: Iterate through each file in folder 'b'
    for filename in os.listdir(folder_b):
        if filename in matched_files_list:
            # Step 4a: Move the file to 'suspect_present' folder
            shutil.move(os.path.join(folder_b, filename), os.path.join(suspect_present_folder, filename))
            #print("Copied one file")
            
        else:
            # Step 4b: Move the file to 'suspect_abscent' folder
            shutil.move(os.path.join(folder_b, filename), os.path.join(suspect_abscent_folder, filename)) 
            
        count_files_movement+=1
        print("Moving Files :",count_files_movement)

    print("Files have been moved based on the matching criteria.")




if __name__ == '__main__':
    main()