import time
import os
import subprocess
import threading

def download_chunk(url, input_dir, chunk_duration):
    download_start_time = time.time()
    if os.path.exists(input_dir):
        pass
    else:
        os.makedirs(input_dir)
    print("Downloading chunks")
    chunk_filename = os.path.join(input_dir, f"chunk_{chunk_counter}.mp4")
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
    subprocess.run(cmd, capture_output=True, text=True)
    print(f"Downloaded chunk {chunk_counter}")
    download_end_time = time.time()
    download_time = download_end_time - download_start_time
    time.sleep(abs(chunk_duration - download_time))

def main():
    url = "https://www.youtube.com/watch?v=sUKwTVAc0Vo"
    chunk_duration = 5
    input_dir = "input_chunks1"
    global chunk_counter
    chunk_counter = 0
    
    while True:
        chunk_counter += 1
        # Start the thread for downloading chunks
        download_thread = threading.Thread(target=download_chunk, args=(url, input_dir, chunk_duration))
        print("Download with threading")
        download_thread.daemon = True
        download_thread.start()
        
        
if __name__=="__main__":
    main()