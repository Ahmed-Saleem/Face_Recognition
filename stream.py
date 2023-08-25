#pytorch
from concurrent.futures import thread
from sqlalchemy import null
import torch
from torchvision import transforms
import time
import os
import subprocess
import threading

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
    
def append_chunk_to_video(video_paths, output_video_path):
    # Create a text file containing the list of video files to concatenate
    video_list = "vidlist.txt"
    with open(video_list, "w") as f:
        for path in video_paths:
            f.write(f"file '{path}'\n")
    
    # Run the FFmpeg command
    ffmpeg_cmd = [
        "ffmpeg",
        "-f", "concat",
        "-safe", "0",
        "-i", video_list,
        "-c", "copy",
        output_video_path
    ]

    subprocess.run(ffmpeg_cmd)
    print("Videos concatenated successfully.")

    # Clean up the temporary video list file
    if os.path.exists(video_list):
        os.remove(video_list)


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

def processing_chunk(input_path, output_without_audio_path):
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
    #start_total_time = time.time()
    start_time = time.time()
    #print("cap", cap.isOpened())
    while cap.isOpened():
        # Capture frame-by-frame
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        #print("input video FPS:", output_fps)
        print('Frame#', frame_count, "of", total_frames, "frames")

        # If it's not the frame interval, use the previous frame's data
        if frame_count % frame_interval != 0 and frame_count != 1:
            for box, label in zip(prev_frame_faces, prev_frame_labels):
                x1, y1, x2, y2 = box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 146, 230), 2)
                t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 2, 2)[0]
                cv2.rectangle(frame, (x1, y1), (x1 + t_size[0], y1 + t_size[1]), (0, 146, 230), -1)
                cv2.putText(frame, label, (x1, y1 + t_size[1]), cv2.FONT_HERSHEY_PLAIN, 2, [255, 255, 255], 2)
            video.write(frame)
            # Calculate and display the FPS
            new_time = time.time()
            fps = 1 / (new_time - start_time)
            start_time = new_time
            fps_label = "FPS: {:.2f}".format(fps)
            print(fps_label)
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
        # # Get the current position of the video capture
        # position_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
        # timestamp_seconds = int(position_ms / 1000)
        # frame_timestamp= time_str(timestamp_seconds)
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
                    # for p in person_data:
                    #     if label == p['name']:
                    #         p['timestamps'].append(frame_timestamp)
                    #         if p['thumbnail'] == None:
                    #             p['thumbnail'] = numpy_array_to_base64(face_image)
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

def download_youtube_chunks(url, input_dir, chunk_duration):
    print("Counter", chunk_counter)
    if os.path.exists(input_dir):
        pass
    else:
        os.makedirs(input_dir)
    print("Downloading chunks")
    chunk_filename = f"input_chunks/chunk_{chunk_counter}.mp4"
    
    cmd = [
        "streamlink",
        url,
        "720p",
        "-o",
        chunk_filename,
        "--hls-duration", chunk_duration,  # Limit the duration to 4 seconds
        "--hls-segment-threads" , "5",
        "--hls-live-edge", "99999",  # Adjust as needed
        "--stream-timeout", "1215",
        "--force"
    ]

    subprocess.run(cmd, capture_output=True, text=True)

    print(f"Downloaded chunk {chunk_counter}")

def processing_thread():
    # Process each chunk

    input_path = os.path.join(input_dir, os.listdir(input_dir)[0])
    extract_audio(input_path, audio_path)
        
    # Process video
    processing_chunk(input_path, output_without_audio_path)

    # Merge audio into output video
    chunk_output_path = os.path.join(output_dir, f"chunk_output.mp4")
    print("output path:", chunk_output_path)
    merge_audio_into_video(output_without_audio_path, audio_path, chunk_output_path)

    delete_file(audio_path)
    delete_file(output_without_audio_path)
    if chunk_counter == 2:
        shutil.move(chunk_output_path, final_output_video_path)
    else:
        shutil.move(final_output_video_path, output_path)
        video_paths = [output_path, chunk_output_path]
        append_chunk_to_video(video_paths, final_output_video_path)
        delete_file(output_path)
    delete_file(input_path)

def main():
     
      if len(os.listdir(input_dir)) < 1:   
        # downloading chunks
        print("Download without threading")
        download_youtube_chunks(url, input_dir, chunk_duration)
        
      else:
        # Start the thread for downloading chunks
        download_thread = threading.Thread(target=download_youtube_chunks, args=(url, input_dir, chunk_duration))
        print("Download with threading")
        download_thread.daemon = True
        download_thread.start()
        
        # Start the thread for processing downloaded chunks
        process_thread = threading.Thread(target=processing_thread)
        print("Process with threading")
        process_thread.start()
        
        # Wait for both threads to finish
        # print("download join")
        # download_thread.join()
        print("process join")
        process_thread.join()
        print("all join")


if __name__=="__main__":
    
    # Read features
    global images_names, images_embs
    images_names, images_embs = read_features()
    print("Read features successful")
    
    chunk_duration = "6"
    # total_chunks = 1 
    input_dir = "input_chunks"
    url = "https://www.youtube.com/watch?v=-k9xQwkwC9g&ab_channel=DunyaNews"
    output_without_audio_path = "output_without_audio.mp4"
    audio_path = "audio.aac"
    output_dir = "output_videos"
    output_path = os.path.join(output_dir, "output.mp4")
    final_output_video_path = os.path.join(output_dir, "final_output.mp4")
    chunk_counter = 0
    delete_mp4_files(output_dir)
    delete_mp4_files(input_dir)
    while(1):
        # if chunk_counter == 50:
        #     chunk_counter = 0
        
        chunk_counter +=1
        start_total_time = time.time()
        main()
        print("Number of processed chunks:", chunk_counter)
        end_total_time = time.time()
        total_time = end_total_time - start_total_time 
        print("Total chunk processing time:", total_time)
        



