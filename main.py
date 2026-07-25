import os
import subprocess
import argparse
import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
from scenedetect import detect, ContentDetector

def extract_frames(input_video, frames_dir):
    """FFmpeg का उपयोग करके वीडियो से फ्रेम्स अलग करना"""
    os.makedirs(frames_dir, exist_ok=True)
    print("🎬 [1/5] Extracting video frames using FFmpeg...")
    cmd = f"ffmpeg -i {input_video} -q:v 2 {frames_dir}/frame_%05d.png -y"
    subprocess.run(cmd, shell=True, check=True)

def detect_keyframes(input_video):
    """PySceneDetect से सीन परिवर्तन और की-फ्रेम्स डिटेक्ट करना"""
    print("🔍 [2/5] Detecting scene keyframes...")
    scene_list = detect(input_video, ContentDetector(threshold=27.0))
    keyframe_indices = [scene[0].frame_num for scene in scene_list]
    if not keyframe_indices or keyframe_indices[0] != 1:
        keyframe_indices.insert(0, 1)
    print(f"✅ Found {len(keyframe_indices)} Keyframes: {keyframe_indices}")
    return set(keyframe_indices)

def process_frame_to_real_4k(image_path, is_keyframe=False):
    """
    CPU-Optimized Real & 4K Style Filter:
    - कार्टून/एनिमे टेक्सचर को स्मूथ करके स्किन व रियल डिटेल्स उभारना
    - 4K Resolution (3840x2160) में सुपर-सॅम्पलिंग करना
    """
    img = Image.open(image_path).convert("RGB")
    
    # 1. Edge-aware texture smoothing & Realistic detail enhancement
    img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    img_smooth = cv2.edgePreservingFilter(img_cv, flags=1, sigma_s=60, sigma_r=0.4)
    img_detail = cv2.detailEnhance(img_smooth, sigma_s=10, sigma_r=0.15)
    
    result_img = Image.fromarray(cv2.cvtColor(img_detail, cv2.COLOR_BGR2RGB))
    
    # 2. Color & Contrast Balancing for realistic look
    contrast = ImageEnhance.Contrast(result_img).enhance(1.15)
    color = ImageEnhance.Color(contrast).enhance(0.95)
    sharpness = ImageEnhance.Sharpness(color).enhance(1.3)
    
    # 3. 4K High Resolution Upscaling (3840 x 2160)
    final_4k = sharpness.resize((3840, 2160), Image.Resampling.LANCZOS)
    final_4k.save(image_path)

def process_all_frames(frames_dir, keyframes):
    """सारे फ्रेम्स पर पाइपलाइन अप्लाई करना"""
    print("✨ [3/5] Processing frames to Real/4K Style...")
    frame_files = sorted([f for f in os.listdir(frames_dir) if f.endswith('.png')])
    
    total = len(frame_files)
    for idx, fname in enumerate(frame_files, start=1):
        fpath = os.path.join(frames_dir, fname)
        is_key = idx in keyframes
        process_frame_to_real_4k(fpath, is_keyframe=is_key)
        
        if idx % 30 == 0 or idx == total:
            print(f"   Progress: {idx}/{total} frames processed.")

def assemble_video(frames_dir, original_video, output_video):
    """सारे फ्रेम्स को जोड़कर ओरिजिनल ऑडियो के साथ 4K वीडियो बनाना"""
    print("🎞️ [4/5] Reassembling 4K Video with Audio...")
    
    # ओरिजिनल वीडियो का FPS निकालना
    cap = cv2.VideoCapture(original_video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    cap.release()

    temp_video = "temp_render.mp4"
    cmd_stitch = f"ffmpeg -framerate {fps} -i {frames_dir}/frame_%05d.png -c:v libx264 -pix_fmt yuv420p {temp_video} -y"
    subprocess.run(cmd_stitch, shell=True, check=True)

    print("🔊 [5/5] Merging original audio track...")
    cmd_audio = f"ffmpeg -i {temp_video} -i {original_video} -c:v copy -c:a aac -map 0:v:0 -map 1:a:0? {output_video} -y"
    subprocess.run(cmd_audio, shell=True, check=True)

    if os.path.exists(temp_video):
        os.remove(temp_video)
    print(f"🎉 Success! Final 4K Video saved as: {output_video}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cartoon to Real 4K Video Demo")
    parser.add_argument("--input", default="input.mp4", help="Path to input cartoon video")
    parser.add_argument("--output", default="output_4k.mp4", help="Path for output video")
    args = parser.parse_args()

    frames_directory = "extracted_frames"

    if not os.path.exists(args.input):
        raise FileNotFoundError(f"❌ Input video '{args.input}' not found. Please place 'input.mp4' in root folder.")

    # Execute Pipeline
    extract_frames(args.input, frames_directory)
    keyframes = detect_keyframes(args.input)
    process_all_frames(frames_directory, keyframes)
    assemble_video(frames_directory, args.input, args.output)

