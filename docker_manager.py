import tkinter as tk
from tkinter import filedialog
import subprocess
import asyncio
import threading
import os
# Function to log messages to the Tkinter Text widget
def log_message(message, color="black"):
    log_output.insert(tk.END, f"{message}\n")
    log_output.tag_add("log", "1.0", tk.END)
    log_output.tag_configure("log", foreground=color)
    log_output.yview(tk.END)  # Scroll to the end of the Text widget

# Function to handle running the docker container and displaying logs
async def run_docker_container(image_name):
    # Get the selected folder paths
    criminal_folder = criminal_folder_entry.get()
    all_folder = all_folder_entry.get()
    output_folder = output_folder_entry.get()

    if not criminal_folder or not all_folder or not output_folder:
        log_message("Please select all folders: Criminal, All, and Output.", color="red")
        return

    # Construct the docker command to mount the folders and run the app
    docker_command = [
        "docker", "run", "-it", "--rm",
        "-v", f"{criminal_folder}:/app/img/document/a",  # Mount criminal image folder
        "-v", f"{all_folder}:/app/img/document/b",  # Mount all image folder
        "-v", f"{output_folder}:/app/output",  # Mount output folder
        image_name  # Use the image name provided by the user
    ]

    try:
        log_message(f"Running Docker container with image '{image_name}'...", color="blue")

        # Run the Docker container and capture the output (stdout and stderr)
        process = await asyncio.create_subprocess_exec(*docker_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=False)

        # Async function to read the output and error streams in real-time
        async def read_output():
            # Read stdout and stderr asynchronously
            while True:
                stdout_line = await process.stdout.readline()
                stderr_line = await process.stderr.readline()

                if stdout_line:
                    log_message(stdout_line.decode().strip(), color="blue")  # Log standard output from the container
                if stderr_line:
                    log_message(stderr_line.decode().strip(), color="red")  # Log error output from the container

                if not stdout_line and not stderr_line and process.returncode is not None:
                    break

            log_message("Finished reading Docker output.", color="green")

        # Run the log reading function asynchronously
        await read_output()

        # Wait for the process to finish
        await process.wait()
        log_message("Docker container finished running!", color="green")
        # Log the list of files in folders a and b
       
    
    except Exception as e:
        log_message(f"Failed to run Docker container: {e}", color="red")
    
    except Exception as e:
        log_message(f"Failed to run Docker container: {e}", color="red")

        

# Function to check if the Docker image exists, and build it if necessary
async def check_and_run_docker(image_name, dockerfile_path='.'):
    # Check if the Docker image exists locally
    try:
        log_message("Checking if Docker image exists...", color="blue")
        
        # Check for the Docker image
        result = await asyncio.create_subprocess_exec("docker", "images", "-q", image_name, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=False)
        stdout, stderr = await result.communicate()

        if not stdout.strip():
            log_message(f"Image '{image_name}' not found, building from Dockerfile...", color="red")
            
            # Build the Docker image from Dockerfile
            build_result = await asyncio.create_subprocess_exec("docker", "build", "-t", image_name, dockerfile_path, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=False)
            build_stdout, build_stderr = await build_result.communicate()

            if build_stdout:
                log_message(build_stdout.decode(), color="blue")
            if build_stderr:
                log_message(build_stderr.decode(), color="red")
            
            if build_result.returncode == 0:
                log_message(f"Image '{image_name}' built successfully!", color="green")
            else:
                log_message(f"Error building the image '{image_name}'", color="red")
                return
        else:
            log_message(f"Image '{image_name}' already exists.", color="green")
        
        # Run the Docker container after ensuring the image is present
        await run_docker_container(image_name)
        
    except Exception as e:
        log_message(f"Error checking/building the image: {e}", color="red")

# Function to select the criminal images folder
def select_criminal_folder():
    folder_selected = filedialog.askdirectory()
    criminal_folder_entry.delete(0, tk.END)
    criminal_folder_entry.insert(0, folder_selected if folder_selected.endswith('/') else folder_selected + '/')

# Function to select the all images folder
def select_all_folder():
    folder_selected = filedialog.askdirectory()
    all_folder_entry.delete(0, tk.END)
    all_folder_entry.insert(0, folder_selected if folder_selected.endswith('/') else folder_selected + '/')

# Function to select the output folder
def select_output_folder():
    folder_selected = filedialog.askdirectory()
    output_folder_entry.delete(0, tk.END)
    output_folder_entry.insert(0, folder_selected if folder_selected.endswith('/') else folder_selected + '/')

# Function to handle the input and run the Docker container in a separate thread
def run():
    image_name = docker_image_name_entry.get()
    if not image_name:
        log_message("Please enter the Docker image name.", color="red")
        return
    threading.Thread(target=asyncio.run, args=(check_and_run_docker(image_name),), daemon=True).start()

# Create the main Tkinter window
root = tk.Tk()
root.title("Docker Image Runner")

# Create UI components
tk.Label(root, text="Enter Docker Image Name:").pack(pady=10)
docker_image_name_entry = tk.Entry(root, width=40)
docker_image_name_entry.pack(pady=10)

tk.Button(root, text="Select Criminal Image Folder", command=select_criminal_folder).pack(pady=5)
criminal_folder_entry = tk.Entry(root, width=40)
criminal_folder_entry.pack(pady=5)

tk.Button(root, text="Select All Image Folder", command=select_all_folder).pack(pady=5)
all_folder_entry = tk.Entry(root, width=40)
all_folder_entry.pack(pady=5)

tk.Button(root, text="Select Output Folder", command=select_output_folder).pack(pady=5)
output_folder_entry = tk.Entry(root, width=40)
output_folder_entry.pack(pady=5)

tk.Button(root, text="Run Docker Container", command=run).pack(pady=10)

# Create a text widget to display the logs
log_output = tk.Text(root, height=15, width=80)
log_output.pack(pady=10)

# Start the Tkinter event loop
root.mainloop()
