import os
import shutil
import logging

# Set up logging configuration
logging.basicConfig(level=logging.INFO, format='%(message)s', handlers=[logging.StreamHandler()])
logger = logging.getLogger()

def main():
    # Define the paths for the folders
    output_folder = 'output'
    folder_b = 'img/document/b'
    suspect_present_folder = 'suspect_present'
    suspect_absent_folder = 'suspect_absent'

    logger.info("Starting the process...")

    # Step 1: Read all text files in the 'output' folder and store file names into a list
    matched_files_list = []
    logger.info(f"Checking if '{output_folder}' exists...")
    if not os.path.exists(output_folder):
        logger.error(f"Output folder '{output_folder}' does not exist!")
        return

    logger.info(f"Scanning for .txt files in '{output_folder}'...")
    try:
        files_in_output = os.listdir(output_folder)
        logger.info(f"Files found in '{output_folder}': {files_in_output}")
        for filename in files_in_output:
            if filename.endswith(".txt"):
                file_path = os.path.join(output_folder, filename)
                logger.info(f"Reading file: {file_path}")
                with open(file_path, 'r') as file:
                    # Add the file content (assuming one file name per line)
                    matched_files_list.extend(file.read().splitlines())
                logger.info(f"Reading file: {file_path}")
                with open(file_path, 'r') as file:
                    # Add the file content (assuming one file name per line)
                    matched_files_list.extend(file.read().splitlines())
        logger.info(f"Matched files list: {matched_files_list}")
    except Exception as e:
        logger.error(f"Error reading files from {output_folder}: {e}")

    if not matched_files_list:
        logger.warning("No matched files found in the output folder!")

    # Step 2: Ensure that the suspect folders exist
    logger.info(f"Ensuring suspect folders '{suspect_present_folder}' and '{suspect_absent_folder}' exist...")
    try:
        os.makedirs(suspect_present_folder, exist_ok=True)
        os.makedirs(suspect_absent_folder, exist_ok=True)
        logger.info("Suspect folders are ready.")
    except Exception as e:
        logger.error(f"Error creating suspect folders: {e}")

    count_files_movement = 0
    logger.info(f"Processing files in folder '{folder_b}'...")

    # Step 3: Iterate through each file in folder 'b'
    try:
        for filename in os.listdir(folder_b):
            file_path_b = os.path.join(folder_b, filename)
            logger.info(f"Processing file: {filename}")

            # Step 4a: Move the file if it exists in the matched list
            if filename in matched_files_list:
                logger.info(f"File '{filename}' matched. Moving to suspect_present folder.")
                try:
                    shutil.move(file_path_b, os.path.join(suspect_absent_folder, filename))
                    logger.info(f"Moved file {filename} to '{suspect_absent_folder}'")
                except Exception as e:
                    logger.error(f"Error moving file '{filename}': {e}")
            else:
                # Step 4b: Move the file if it does not exist in the matched list
                logger.info(f"File '{filename}' not matched. Moving to suspect_absent folder.")
                try:
                    shutil.move(file_path_b, os.path.join(suspect_abscent_folder, filename))
                    logger.info(f"Moved file {filename} to '{suspect_abscent_folder}'")
                except Exception as e:
                    logger.error(f"Error moving file '{filename}': {e}")

            count_files_movement += 1
            logger.info(f"Files processed so far: {count_files_movement}")

        logger.info("File processing completed.")
    except Exception as e:
        logger.error(f"Error processing files in {folder_b}: {e}")

    # Step 5: Summarize
    try:
        total_files_in_b = len(os.listdir(folder_b))
        logger.info(f"Total files in '{folder_b}': {total_files_in_b}")
        logger.info(f"Total files moved: {count_files_movement}")
        logger.info(f"Files have been moved based on the matching criteria.")
    except Exception as e:
        logger.error(f"Error summarizing the file counts: {e}")

if __name__ == '__main__':
    main()
