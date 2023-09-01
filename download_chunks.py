import os
from datetime import datetime
import time
import subprocess
import pandas as pd

def download_chunk(url, chunk_filename, chunk_duration, chunk_counter):
    download_start_time = time.time()
    print("Downloading chunks")
    cmd = [
        "streamlink",
        "--hls-duration", str(chunk_duration),
        "-o",
        chunk_filename,
        url,
        "best",
        "--hls-segment-threads" , "5",
        "--hls-live-edge", "99999",
        "--stream-timeout", "1215",
        "--force"
    ]
    current_time = datetime.now().time().strftime('%H:%M:%S')
    subprocess.run(cmd, capture_output=True, text=True)
    print(f"Downloaded chunk {chunk_counter}")
    download_end_time = time.time()
    download_time = download_end_time - download_start_time
    print("Chunk downloading time: ", download_time)
    input_time_dict['filename'] = chunk_filename
    input_time_dict['download_timestamp'] = current_time
    input_time_list.append(input_time_dict)
        
    # Convert list of dictionaries to DataFrame
    df = pd.DataFrame(input_time_list)

    # Save DataFrame to CSV file
    df.to_csv(input_time_list_path, index=False)
    print(f"DataFrame saved to {input_time_list_path}.")
    if download_time < chunk_duration:
        time.sleep(chunk_duration - download_time)
    
    
        
def delete_mp4_files(directory):
    for filename in os.listdir(directory):
        if filename.endswith(".mp4"):
            file_path = os.path.join(directory, filename)
            os.remove(file_path)
            print(f"Deleted: {file_path}")

def delete_file(file_path):
    file_path = file_path
    if os.path.exists(file_path):
        os.remove(file_path)
        print(f"File '{file_path}' has been deleted.")
    else:
        print(f"File '{file_path}' does not exist.")
            
def main():
    global input_time_dict, input_time_list, input_time_list_path
    input_dir = "input_chunks"
    delete_mp4_files(input_dir)
    # list of download timestamps
    input_time_dict = {
            'filename' : '',
            'download_timestamp' : '00:00:00'
    }
    input_time_list = []
    
    url = "https://www.youtube.com/watch?v=sUKwTVAc0Vo"
    playback_duration = 1                       #playback duration in hours
    chunk_duration = 10
    chunk_counter = 0
    chunk_threshold = int((playback_duration*3600)/chunk_duration)
    if not os.path.exists(input_dir):
        os.mkdir(input_dir)
    input_time_list_path = os.path.join(input_dir, "input_time_list.csv")
        
    while(1):
        chunk_filename = os.path.join(input_dir, f"chunk_{chunk_counter}.mp4")
        download_chunk(url, chunk_filename, chunk_duration, chunk_counter)
        # print("Download timestamp: ", current_time)        
        chunk_counter += 1
        
        if chunk_counter == chunk_threshold:
            delete_mp4_files(input_dir)
            delete_file(input_time_list_path)
            input_time_list = []
            chunk_counter = 0
        
            
if __name__=="__main__":
    main()
        
    
    