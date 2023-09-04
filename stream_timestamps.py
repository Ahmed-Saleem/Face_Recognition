#pytorch
from concurrent.futures import thread
from sqlalchemy import null
import torch
from torchvision import transforms
import time
import os
import subprocess
import threading
import queue
import pafy
import ffmpeg
import pytz

# Set the time zone to Pakistan Standard Time (PKT)
pakistan_timezone = pytz.timezone('Asia/Karachi')

#other lib
import shutil
import sys
import numpy as np
import base64 
import cv2
import pandas as pd
from datetime import datetime
import json
import glob
import shutil
#from moviepy.editor import VideoFileClip, AudioFileClip
#os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

#sys.path.insert(0, "scripts/yolov5_face")
sys.path.insert(0, "yolov5_face")
from models.experimental import attempt_load
from utils.datasets import letterbox
from utils.general import check_img_size, non_max_suppression_face, scale_coords

# Check device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Variables to store the detected faces and labels for the previous fully processed frame
prev_frame_faces = []
prev_frame_labels = []
person_data = []

#model = attempt_load("scripts/yolov5_face/yolov5m-face.pt", map_location=device)
model = attempt_load("yolov5_face/yolov5m-face.pt", map_location=device)

# Get model recognition 
from insightface.insight_face import iresnet100
#weight = torch.load("scripts/insightface/resnet100_backbone.pth", map_location = device)
weight = torch.load("insightface/resnet100_backbone.pth", map_location = device)
model_emb = iresnet100()
model_emb.load_state_dict(weight)
model_emb.to(device)
model_emb.eval()

face_preprocess = transforms.Compose([
                                    transforms.ToTensor(), # input PIL => (3,56,56), /255.0
                                    transforms.Resize((112, 112)),
                                    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
                                    ])

# isThread = True
score = 0
name = null


# Resize image
def resize_image(img0, img_size):
    h0, w0 = img0.shape[:2]  # orig hw
    r = img_size / max(h0, w0)  # resize image to img_size

    if r != 1:  # always resize down, only resize up if training with augmentation
        interp = cv2.INTER_AREA if r < 1  else cv2.INTER_LINEAR
        img0 = cv2.resize(img0, (int(w0 * r), int(h0 * r)), interpolation=interp)

    imgsz = check_img_size(img_size, s=model.stride.max())  # check img_size
    img = letterbox(img0, new_shape=imgsz)[0]

    # Convert
    img = img[:, :, ::-1].transpose(2, 0, 1).copy()  # BGR to RGB, to 3x416x416

    img = torch.from_numpy(img).to(device)
    img = img.float()  # uint8 to fp16/32
    img /= 255.0  # 0 - 255 to 0.0 - 1.0
    
    return img

def scale_coords_landmarks(img1_shape, coords, img0_shape, ratio_pad=None):
    # Rescale coords (xyxy) from img1_shape to img0_shape
    if ratio_pad is None:  # calculate from img0_shape
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  # gain  = old / new
        pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2  # wh padding
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]

    coords[:, [0, 2, 4, 6, 8]] -= pad[0]  # x padding
    coords[:, [1, 3, 5, 7, 9]] -= pad[1]  # y padding
    coords[:, :10] /= gain
    #clip_coords(coords, img0_shape)
    coords[:, 0].clamp_(0, img0_shape[1])  # x1
    coords[:, 1].clamp_(0, img0_shape[0])  # y1
    coords[:, 2].clamp_(0, img0_shape[1])  # x2
    coords[:, 3].clamp_(0, img0_shape[0])  # y2
    coords[:, 4].clamp_(0, img0_shape[1])  # x3
    coords[:, 5].clamp_(0, img0_shape[0])  # y3
    coords[:, 6].clamp_(0, img0_shape[1])  # x4
    coords[:, 7].clamp_(0, img0_shape[0])  # y4
    coords[:, 8].clamp_(0, img0_shape[1])  # x5
    coords[:, 9].clamp_(0, img0_shape[0])  # y5
    return coords

def get_face(input_image):
    # Parameters
    size_convert = 704
    conf_thres = 0.75
    iou_thres = 0.75
    
    # Resize image
    img = resize_image(input_image.copy(), size_convert)

    # Via yolov5-face
    with torch.no_grad():
        pred = model(img[None, :])[0]

    # Apply NMS
    det = non_max_suppression_face(pred, conf_thres, iou_thres)[0]
    bboxs = np.int32(scale_coords(img.shape[1:], det[:, :4], input_image.shape).round().cpu().numpy())
    landmarks = np.int32(scale_coords_landmarks(img.shape[1:], det[:, 5:15], input_image.shape).round().cpu().numpy())    
    
    return bboxs, landmarks

def get_feature(face_image, training = True): 
    # Convert to RGB
    face_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
    
    # Preprocessing image BGR
    face_image = face_preprocess(face_image).to(device)
    
    # Via model to get feature
    with torch.no_grad():
        if training:
            emb_img_face = model_emb(face_image[None, :])[0].cpu().numpy()
        else:
            emb_img_face = model_emb(face_image[None, :]).cpu().numpy()
    
    # Convert to array
    images_emb = emb_img_face/np.linalg.norm(emb_img_face)
    return images_emb

#def read_features(root_fearure_path = "scripts/static/feature/face_features.npz"):
def read_features(root_fearure_path = "static/feature/face_features.npz"):
    data = np.load(root_fearure_path, allow_pickle=True)
    images_name = data["arr1"]
    images_emb = data["arr2"]
    
    return images_name, images_emb

def recognition(face_image, images_names, images_embs):
    global isThread, score, name
    
    # Get feature from face
    query_emb = (get_feature(face_image, training=False))

    scores = (query_emb @ images_embs.T)[0]

    id_min = np.argmax(scores)
    score = scores[id_min]
    name = images_names[id_min]
    return name, score

def time_str(total_seconds):
    seconds = total_seconds % 60
    total_minutes = total_seconds // 60
    minutes = total_minutes % 60
    hours = total_minutes // 60
    timestamp_str = f"{hours:02d}:{minutes:02d}:{seconds:02d}"
    return timestamp_str

def time_to_seconds(timestamp_str):
    try:
        hours, minutes, seconds = map(int, timestamp_str.split(":"))
        total_seconds = hours * 3600 + minutes * 60 + seconds
        return total_seconds
    except ValueError:
        raise ValueError("Invalid timestamp format. Use hh:mm:ss")
    
def numpy_array_to_base64(image_array, format='.jpg'):
    _, buffer = cv2.imencode(format, image_array)
    base64_image = base64.b64encode(buffer).decode()
    return base64_image

def extract_audio(video_path, audio_output_path):
    ffmpeg_cmd = [
    'ffmpeg',
    '-y',            
    '-i', video_path,   # Input video file
    '-vn',               # Disable video recording
    '-acodec', 'copy',   # Use the same audio codec as the input
    audio_output_path   # Output audio file
    ]
    subprocess.run(ffmpeg_cmd, check=True)

    
def merge_audio_into_video(video_path, audio_path, output_path):
    ffmpeg_cmd = [
    'ffmpeg',
    '-y',
    '-i', video_path,
    '-i', audio_path,
    '-c:v', 'copy',
    '-map', '0:v:0',
    '-map', '1:a:0',
    '-shortest',
    output_path
    ]
    subprocess.run(ffmpeg_cmd, check=True)
    
# def append_chunk_to_video(video_paths, output_video_path):
#     # Create a text file containing the list of video files to concatenate
#     video_list = "vidlist.txt"
#     with open(video_list, "w") as f:
#         for path in video_paths:
#             f.write(f"file '{path}'\n")
    
#     # Run the FFmpeg command
#     ffmpeg_cmd = [
#         "ffmpeg",
#         "-f", "concat",
#         "-safe", "0",
#         "-i", video_list,
#         "-c", "copy",
#         output_video_path
#     ]

#     subprocess.run(ffmpeg_cmd)
#     print("Videos concatenated successfully.")

#     # Clean up the temporary video list file
#     if os.path.exists(video_list):
#         os.remove(video_list)


def delete_file(file_path):
    file_path = file_path
    if os.path.exists(file_path):
        os.remove(file_path)
        print(f"File '{file_path}' has been deleted.")
    else:
        print(f"File '{file_path}' does not exist.")

def delete_mp4_files(directory):
    for filename in os.listdir(directory):
        if filename.endswith(".mp4"):
            file_path = os.path.join(directory, filename)
            os.remove(file_path)
            print(f"Deleted: {file_path}")

def processing_chunk(input_path, output_without_audio_path, chunk_timestamp):
    global json_data, indices_to_delete
    # Open video
    cap = cv2.VideoCapture(input_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))  # Get total frames of the input video
    frame_count = 0

    # Save video
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    output_fps = cap.get(cv2.CAP_PROP_FPS)
    size = (frame_width, frame_height)
    video = cv2.VideoWriter(output_without_audio_path, cv2.VideoWriter_fourcc(*'mp4v'), output_fps, size)

    # Add frame interval
    frame_interval = int(output_fps / 3)

    # Read until video is completed
    while cap.isOpened():
        # Capture frame-by-frame
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        #print("input video FPS:", output_fps)
        print('Frame#', frame_count, "of", total_frames, "frames")
        # Get the current position of the video capture
        position_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
        timestamp_seconds = int((position_ms / 1000) + time_to_seconds(chunk_timestamp))
        frame_timestamp = time_str(timestamp_seconds)

        # If it's not the frame interval, use the previous frame's data
        if frame_count % frame_interval != 0 and frame_count != 1:
            for box, label in zip(prev_frame_faces, prev_frame_labels):
                x1, y1, x2, y2 = box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 146, 230), 2)
                t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 2, 2)[0]
                cv2.rectangle(frame, (x1, y1), (x1 + t_size[0], y1 + t_size[1]), (0, 146, 230), -1)
                cv2.putText(frame, label, (x1, y1 + t_size[1]), cv2.FONT_HERSHEY_PLAIN, 2, [255, 255, 255], 2)
            video.write(frame)
            # # Calculate and display the FPS
            # new_time = time.time()
            # fps = 1 / (new_time - start_time)
            # start_time = new_time
            # fps_label = "FPS: {:.2f}".format(fps)
            # print(fps_label)
            #cv2.putText(frame, fps_label, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            continue

        # Calculate and display the FPS
        # new_time = time.time()
        # fps = 1 / (new_time - start_time)
        # start_time = new_time
        # fps_label = "FPS: {:.2f}".format(fps)
        # print(fps_label)
        #cv2.putText(frame, fps_label, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # Get faces
        bboxs, landmarks = get_face(frame)
        # h, w, c = frame.shape
        
        # tl = 1 or round(0.002 * (h + w) / 2) + 1  # line/font thickness
        # clors = [(255,0,0),(0,255,0),(0,0,255),(255,255,0),(0,255,255)]
        
        # Get boxs
        prev_frame_faces = []
        prev_frame_labels = []
                        
        for i in range(len(bboxs)):
            # Get location face
            x1, y1, x2, y2 = bboxs[i]
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 146, 230), 2)
            
            # Get recognized name
            face_image = frame[y1:y2, x1:x2]
            name, score = recognition(face_image, images_names, images_embs)
            #print("Detected: ", name, "with score: ", score)
            # # Get recognized name
            # if isThread == True:
            #     isThread = False
                
            #     # Recognition
            #     face_image = frame[y1:y2, x1:x2]
            #     thread = Thread(target=recognition, args=(face_image, images_names, images_embs))
            #     thread.start()

            # # Landmarks
            # for x in range(5):
            #     point_x = int(landmarks[i][2 * x])
            #     point_y = int(landmarks[i][2 * x + 1])
            #     cv2.circle(frame, (point_x, point_y), tl+1, clors[x], -1)

            if name == null:
                continue
            else:
                if score < 0.35:
                    label = "Unknown"
                    prev_frame_labels.append(label)
                    prev_frame_faces.append(bboxs[i])
                    #print("Detected: ", caption, "with score: ", score)
                    t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 2, 2)[0]
                    cv2.rectangle(frame, (x1, y1), (x1 + t_size[0], y1 + t_size[1]), (0, 146, 230), -1)
                    cv2.putText(frame, label, (x1, y1 + t_size[1]), cv2.FONT_HERSHEY_PLAIN, 2, [255, 255, 255], 2)
                else:
                    label = name.replace("_", " ")
                    for p in person_data:
                        if p['name'] == label:
                            p['timestamps'].append(frame_timestamp)
                            if p['thumbnail'] == None:
                                p['thumbnail'] = numpy_array_to_base64(face_image)
                            if len(p['timestamps']) > 1:
                                if time_to_seconds(p['timestamps'][-1]) - time_to_seconds(p['timestamps'][-2]) > 3:
                                    pass
                                else:
                                    p['coverageTime'] = time_str(time_to_seconds(p['coverageTime']) + (time_to_seconds(p['timestamps'][-1]) - time_to_seconds(p['timestamps'][-2])))
                    caption = f"{label}:{score:.2f}"
                    prev_frame_labels.append(label)
                    prev_frame_faces.append(bboxs[i])
                    #print("Detected: ", caption, "with score: ", score)
                    t_size = cv2.getTextSize(caption, cv2.FONT_HERSHEY_PLAIN, 2, 2)[0]
                    cv2.rectangle(frame, (x1, y1), (x1 + t_size[0], y1 + t_size[1]), (0, 146, 230), -1)
                    cv2.putText(frame, caption, (x1, y1 + t_size[1]), cv2.FONT_HERSHEY_PLAIN, 2, [255, 255, 255], 2)
        # Count fps 
        video.write(frame)
        #cv2.imshow("Face Recognition", frame)
        
        # Press Q on keyboard to  exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break  
    
    video.release()
    cap.release()
    cv2.destroyAllWindows()
    cv2.waitKey(0)
    print("Video without audio saved at: ", output_without_audio_path)
    
    # Convert the dictionary to a DataFrame
    df = pd.DataFrame(person_data)
    condition = df['timestamps'].apply(len) >= 10
    filtered_df = df[condition]
    print("data:", filtered_df)
    
    # DataFrame to .json file
    file_path = "output_videos/data.json"
    json_data = filtered_df.to_json(orient='records')
    
    if filtered_df is not None:
        # Write the data to the JSON file
        with open(file_path, "w") as json_file:
            json.dump(json_data, json_file)
        print("Data saved to:", file_path)
        
        # Define the output CSV file path
        output_csv_path = "output_videos/data.csv"

        # Save the DataFrame to a CSV file
        filtered_df_copy = filtered_df.drop('thumbnail', axis=1)
        filtered_df_copy.to_csv(output_csv_path, index=False)
        print(f"DataFrame saved to '{output_csv_path}'.")
        
    # # Remaining data
    # for p in person_data:
    #     if len(p['timestamps']) >= 10:
    #         p['startTime'].append(p['timestamps'][0])
    #         for t in range(0,len(p['timestamps'])):
    #             ts = time_to_seconds(p['timestamps'][t])
    #             prev_ts = time_to_seconds(p['timestamps'][t-1])
    #             if ts - prev_ts >= 2:
    #                 p['startTime'].append(p['timestamps'][t])
    #                 p['endTime'].append(p['timestamps'][t-1])
    #         if len(p['startTime']) != len(p['endTime']):
    #             if len(p['startTime']) > len(p['endTime']):
    #                 p['endTime'].append(p['timestamps'][-1])
    #             else:
    #                 p['endTime'][:-1]
    #         #print('start',len(p['startTime']))
    #         #print('end',len(p['endTime']))
    #         for tt in range(0, len(p['startTime'])):
    #             tts = (time_to_seconds(p['endTime'][tt]) - time_to_seconds(p['startTime'][tt]))
    #             coverage_time = coverage_time + tts
    #         p['coverageTime'] = time_str(coverage_time)
        
    
  
    # # Define the output CSV file path
    # output_csv_path = "output_videos/data.csv"

    # # Save the DataFrame to a CSV file
    # filtered_df.to_csv(output_csv_path, index=False)
    # print(f"DataFrame saved to '{output_csv_path}'.")
    
def download_chunk(url, input_dir, chunk_duration):
    download_start_time = time.time()
    if not os.path.exists(input_dir):
        os.makedirs(input_dir)
          
    chunk_filename = os.path.join(input_dir, f"chunk_{chunk_counter}.mp4")
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
    current_time = datetime.now(pakistan_timezone).time().strftime('%H:%M:%S')
    subprocess.run(cmd, capture_output=True, text=True)
    print(f"Downloaded chunk {chunk_counter}")
    download_end_time = time.time()
    time_dict['filename'] = chunk_filename
    time_dict['download_timestamp'] = current_time
    chunk_queue.put(time_dict)
    download_time = download_end_time - download_start_time
    print("Chunk downloading time: ", download_time)
    if download_time < chunk_duration:
        time.sleep(chunk_duration - download_time)
    elif download_time == chunk_duration:
        time.sleep(chunk_duration)

# def download_chunks_background(url, input_dir, chunk_duration):
#     global chunk_counter, chunk_queue
    
#     # Define a queue for passing chunks between threads
#     chunk_queue = queue.Queue()
#     chunk_counter = 0
#     while True:
#         chunk_counter += 1
#         download_chunk(url, input_dir, chunk_duration)
#         print(f"Downloaded chunk {chunk_counter}")

def run_download_chunks():
    global chunk_queue, time_dict, chunk_counter
    chunk_queue = queue.Queue()
    time_dict = {
            'filename' : '',
            'download_timestamp' : '00:00:00'
    }
    chunk_counter = 0
    url = "https://www.youtube.com/watch?v=sUKwTVAc0Vo"
    input_dir = "input_chunks"
    playback_duration = 6                       #playback duration in hours
    chunk_duration = 10
    chunk_threshold = int((playback_duration*3600)/chunk_duration)
    chunk_threshold = int((playback_duration*3600)/chunk_duration)
    if not os.path.exists(input_dir):
        os.mkdir(input_dir)
        
    while(1):
        download_chunk(url, input_dir, chunk_duration)
        # print("Download timestamp: ", current_time)        
        chunk_counter += 1
        
        if chunk_counter == chunk_threshold:
            delete_mp4_files(input_dir)
            chunk_counter = 0
            while not chunk_queue.empty():
                item = chunk_queue.get()
                print("Dequeued item:", item)
            print("Queue is now empty")

            
def main():
    
    global images_names, images_embs
    
    # Read features
    images_names, images_embs = read_features()
    print("Read features successful")

    # create list of timestamps
    label_names = list(set(images_names))
    for n in label_names:
        n = n.replace("_", " ")
        person_entry = {
            'thumbnail': None,
            'name': n,
            'timestamps': [],
            'coverageTime': '00:00:00'
        }
        # Append the dictionary to the list
        person_data.append(person_entry)
    
    output_without_audio_path = "output_without_audio.mp4"
    audio_path = "audio.aac"
    output_dir = "output_videos"
    output_chunk_counter = 0
    # output_path = os.path.join(output_dir, "output.mp4")
    # final_output_video_path = os.path.join(output_dir, "final_output.mp4")
     
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    
    # Start the download_chunks.py thread as a daemon
    download_thread = threading.Thread(target=run_download_chunks, daemon=True)
    download_thread.start()
    
    while(1):
        if not chunk_queue.empty():
            input_time_dict = chunk_queue.get()
            input_path = input_time_dict['filename']
            chunk_timestamp = input_time_dict['download_timestamp']
            chunk_queue.task_done()
            processing_start = time.time()
            
            # Extract audio
            extract_audio(input_path, audio_path)
            
            # Process video
            processing_chunk(input_path, output_without_audio_path, chunk_timestamp)

            # Merge audio into output video
            chunk_output_path = os.path.join(output_dir, f"chunk_output_{output_chunk_counter}.mp4")
            print("output path:", chunk_output_path)
            merge_audio_into_video(output_without_audio_path, audio_path, chunk_output_path)

            delete_file(audio_path)
            delete_file(output_without_audio_path)

            processing_end = time.time()
            total_processing_time = processing_end - processing_start
            print("Chunk processing time: ", total_processing_time)
            output_chunk_counter += 1
        else:
            continue
    
    # # Append output video
    # if chunk_counter == 2:
    #     shutil.move(chunk_output_path, final_output_video_path)
    # else:
    #     shutil.move(final_output_video_path, output_path)
    #     video_paths = [output_path, chunk_output_path]
    #     append_chunk_to_video(video_paths, final_output_video_path)
    #     delete_file(output_path)



if __name__=="__main__":
    main()

# if __name__=="__main__":
#     while(1):
#         chunk_counter +=1
#         start_total_time = time.time()
#         if len(os.listdir(input_dir)) < 1:   
#             # downloading chunks
#             print("Download without threading")
#             download_chunk(url, input_dir, chunk_duration)
#         else:
#             main()
#         print("Number of processed chunks:", chunk_counter)
#         end_total_time = time.time()
#         total_time = end_total_time - start_total_time 
#         print("Total chunk time:", total_time)
#         # if chunk_counter % 4320 == 0:
#         #     delete_mp4_files(output_dir)
        



