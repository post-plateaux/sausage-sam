import os
import sys
import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
import gc
import shutil
import traceback
import concurrent.futures
from pathlib import Path
import base64
from flask_socketio import SocketIO
from collections import OrderedDict
import imageio

# Add the segment-anything-2 directory to the Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
segment_anything_2_dir = os.path.join(current_dir, 'segment-anything-2')
sys.path.append(segment_anything_2_dir)

from sam2.build_sam import build_sam2_video_predictor

# Set environment variable for CuDNN backend
os.environ['TORCH_CUDNN_SDPA_ENABLED'] = '1'

# Editable parameters
INPUT_VIDEO = "input/dijon-raft.mp4"
SAM2_CHECKPOINT = "segment-anything-2/checkpoints/sam2_hiera_tiny.pt"
MODEL_CFG = "sam2_hiera_t.yaml"
TEST_MODE = True  # Set to True to process only a short duration
TEST_DURATION = 5  # Duration in seconds for test mode
BATCH_SIZE = 250  # Number of frames to process in each batch

def initialize_sam2():
    predictor = build_sam2_video_predictor(MODEL_CFG, SAM2_CHECKPOINT, device="cuda")
    return predictor

def check_frames_extracted(output_dir):
    if not os.path.exists(output_dir):
        return False
    frame_files = [f for f in os.listdir(output_dir) if f.endswith('.jpg')]
    return len(frame_files) > 0

def extract_frames(input_video, output_dir, progress_callback=None):
    if check_frames_extracted(output_dir):
        print("Frames already extracted. Skipping extraction.")
        cap = cv2.VideoCapture(input_video)
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = len([f for f in os.listdir(output_dir) if f.endswith('.jpg')])
        cap.release()
        if progress_callback:
            progress_callback(20, f"Frames already extracted: {frame_count}")
        return frame_count, fps

    cap = cv2.VideoCapture(input_video)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_count = 0

    os.makedirs(output_dir, exist_ok=True)

    update_interval = max(1, total_frames // 100)  # Update progress every 1% or every frame, whichever is larger

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_path = os.path.join(output_dir, f"{frame_count:05d}.jpg")
        cv2.imwrite(frame_path, frame)
        frame_count += 1

        if frame_count % update_interval == 0 or frame_count == total_frames:
            progress = (frame_count / total_frames) * 20  # 20% of total progress
            if progress_callback:
                progress_callback(progress, f"Extracted {frame_count}/{total_frames} frames")

    cap.release()
    if progress_callback:
        progress_callback(20, f"Extracted {frame_count} frames")
    return frame_count, fps

def get_user_input(frame):
    # Convert frame to base64 string for sending to client
    _, buffer = cv2.imencode('.jpg', frame)
    frame_base64 = base64.b64encode(buffer).decode('utf-8')
    
    # This function will now return the base64 encoded image
    # The actual user input will be handled by the client-side JavaScript
    return frame_base64

def process_frame(frame, mask_logits, green_screen, frame_idx, masks_dir, socketio=None):
    try:
        mask = (mask_logits[0] > 0.0).cpu().numpy().astype(np.uint8)
        
        if mask.shape[:2] != frame.shape[:2]:
            if mask.ndim == 3 and mask.shape[0] == 1:
                mask = mask.squeeze(0)
            mask = cv2.resize(mask, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_NEAREST)

        mask = (mask * 255).astype(np.uint8)  # Scale mask to 0-255 range
        
        # Save the mask
        mask_path = os.path.join(masks_dir, f"{frame_idx:05d}.png")
        cv2.imwrite(mask_path, mask)
        
        # Apply the mask to the original frame
        masked_frame = cv2.bitwise_and(frame, frame, mask=mask)
        
        # Invert the mask for the green screen
        inverted_mask = cv2.bitwise_not(mask)
        
        # Apply the inverted mask to the green screen
        masked_green_screen = cv2.bitwise_and(green_screen, green_screen, mask=inverted_mask)
        
        # Combine the masked frame and the masked green screen
        result = cv2.add(masked_frame, masked_green_screen)
        
        # Encode the mask and frame for preview
        _, mask_buffer = cv2.imencode('.png', mask)
        mask_base64 = base64.b64encode(mask_buffer).decode('utf-8')
        
        _, frame_buffer = cv2.imencode('.jpg', result)
        frame_base64 = base64.b64encode(frame_buffer).decode('utf-8')
        
        # Send the preview to the client
        if socketio:
            socketio.emit('mask_preview', {'mask': mask_base64, 'frame': frame_base64, 'frame_idx': frame_idx})
        
        return result, mask, mask_base64
    except Exception as e:
        print(f"Error processing frame: {str(e)}")
        traceback.print_exc()
        return None, None, None

def process_batch(predictor, batch_dir, inference_state, input_points, input_labels, green_screen, masks_dir, first_batch=False, progress_callback=None, batch_start_progress=0, batch_end_progress=0, socketio=None):
    try:
        if first_batch:
            predictor.add_new_points_or_box(
                inference_state=inference_state,
                frame_idx=0,
                obj_id=1,
                points=input_points,
                labels=input_labels,
            )

        last_mask = None
        processed_frames = {}
        total_frames = len([f for f in os.listdir(batch_dir) if f.endswith('.jpg')])
        with concurrent.futures.ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
            future_to_frame = {}
            for i, (out_frame_idx, out_obj_ids, out_mask_logits) in enumerate(predictor.propagate_in_video(inference_state)):
                frame_path = os.path.join(batch_dir, f"{out_frame_idx:05d}.jpg")
                frame = cv2.imread(frame_path)
                if frame is None:
                    print(f"Failed to read frame: {frame_path}")
                    continue
                future = executor.submit(process_frame, frame, out_mask_logits, green_screen, out_frame_idx, masks_dir, socketio)
                future_to_frame[future] = out_frame_idx
                
                # Update progress
                progress = batch_start_progress + (i / total_frames) * (batch_end_progress - batch_start_progress)
                if progress_callback:
                    progress_callback(progress, f"Propagating frame {i+1}/{total_frames} in current batch")

            for future in concurrent.futures.as_completed(future_to_frame):
                out_frame_idx = future_to_frame[future]
                result, mask, mask_base64 = future.result()
                if result is not None:
                    processed_frames[out_frame_idx] = result
                    last_mask = mask
                    # Send mask preview to the client
                    if socketio:
                        socketio.emit('mask_preview', {'mask': mask_base64, 'frame_idx': out_frame_idx})

        return processed_frames, last_mask
    except Exception as e:
        print(f"Error in process_batch: {str(e)}")
        traceback.print_exc()
        return None, None

def process_video(predictor, frames_dir, output_video, output_gif, fps, input_points, input_labels, test_mode=False, test_duration=5, batch_size=BATCH_SIZE, progress_callback=None, socketio=None):
    try:
        frame_names = sorted([f for f in os.listdir(frames_dir) if f.endswith('.jpg')])
        first_frame = cv2.imread(os.path.join(frames_dir, frame_names[0]))
        height, width = first_frame.shape[:2]

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Try 'mp4v' codec instead of 'avc1'
        out = cv2.VideoWriter(output_video, fourcc, fps, (width, height))
        if not out.isOpened():
            error_message = f"Could not open video writer for {output_video}"
            print(error_message)
            print(f"Codec: {fourcc}, FPS: {fps}, Size: {width}x{height}")
            print(f"OpenCV version: {cv2.__version__}")
            print(f"Output directory exists: {os.path.exists(os.path.dirname(output_video))}")
            print(f"Output directory writable: {os.access(os.path.dirname(output_video), os.W_OK)}")
            raise Exception(error_message)
        print(f"Video writer opened with codec: {fourcc}, FPS: {fps}, size: {width}x{height}")

        if progress_callback:
            progress_callback(25, "Processing video with user input")

        # Create a green screen background
        green_screen = np.full((height, width, 3), (0, 255, 0), dtype=np.uint8)

        # Create a directory for masks
        masks_dir = os.path.join(os.path.dirname(frames_dir), "masks")
        os.makedirs(masks_dir, exist_ok=True)

        # Process video in batches
        last_mask = None
        max_frames = int(test_duration * fps) if test_mode else len(frame_names)
        total_batches = (max_frames + batch_size - 1) // batch_size

        all_processed_frames = OrderedDict()

        for i in range(0, max_frames, batch_size):
            batch_frame_names = frame_names[i:min(i+batch_size, max_frames)]
            
            # Create a temporary directory for the current batch
            batch_dir = os.path.join(frames_dir, f"batch_{i}")
            os.makedirs(batch_dir, exist_ok=True)
            
            # Copy frames for the current batch to the temporary directory
            for j, frame_name in enumerate(batch_frame_names):
                src = os.path.join(frames_dir, frame_name)
                dst = os.path.join(batch_dir, f"{j:05d}.jpg")
                shutil.copy(src, dst)
                progress = 30 + (i / max_frames) * 10 + (j / len(batch_frame_names)) * (10 / total_batches)
                if progress_callback:
                    progress_callback(progress, f"Copying frame {j+1}/{len(batch_frame_names)} for batch {i // batch_size + 1}/{total_batches}")
            
            # Initialize state for the current batch
            if progress_callback:
                progress_callback(40 + (i / max_frames) * 10, f"Initializing state for batch {i // batch_size + 1}/{total_batches}")
            inference_state = predictor.init_state(video_path=batch_dir)
            
            # Calculate progress range for this batch
            batch_start_progress = 40 + (i / max_frames) * 50
            batch_end_progress = 40 + ((i + batch_size) / max_frames) * 50
            
            if i == 0:
                # Process first batch
                processed_frames, last_mask = process_batch(predictor, batch_dir, inference_state, input_points, input_labels, green_screen, masks_dir, first_batch=True, progress_callback=progress_callback, batch_start_progress=batch_start_progress, batch_end_progress=batch_end_progress, socketio=socketio)
            else:
                # Use the last mask from the previous batch as the starting point
                if last_mask is not None:
                    predictor.add_new_mask(
                        inference_state=inference_state,
                        frame_idx=0,
                        obj_id=1,
                        mask=last_mask,
                    )
                processed_frames, last_mask = process_batch(predictor, batch_dir, inference_state, input_points, input_labels, green_screen, masks_dir, progress_callback=progress_callback, batch_start_progress=batch_start_progress, batch_end_progress=batch_end_progress, socketio=socketio)

            # Add processed frames to the OrderedDict
            for frame_idx, frame in processed_frames.items():
                all_processed_frames[i + frame_idx] = frame

            # Clear CUDA cache after each batch
            torch.cuda.empty_cache()
            gc.collect()
            
            # Remove the temporary batch directory
            shutil.rmtree(batch_dir)

            if test_mode and i + batch_size >= max_frames:
                status = "Test mode: Stopping after processing the specified duration."
                if progress_callback:
                    progress_callback(90, status)
                break

        # Write all processed frames in order
        sorted_frame_indices = sorted(all_processed_frames.keys())
        for j, frame_idx in enumerate(sorted_frame_indices):
            out.write(all_processed_frames[frame_idx])
            progress = 90 + (j / len(sorted_frame_indices)) * 10
            if progress_callback:
                progress_callback(progress, f"Writing frame {j+1}/{len(sorted_frame_indices)}")

        out.release()

        # Create GIF from processed frames
        gif_frames = []
        for frame_idx in sorted_frame_indices:
            frame = cv2.cvtColor(all_processed_frames[frame_idx], cv2.COLOR_BGR2RGB)
            # Resize the frame to make the GIF smaller
            frame = cv2.resize(frame, (320, 180), interpolation=cv2.INTER_AREA)
            gif_frames.append(frame)
        
        # Save the GIF with loop=0 to make it loop indefinitely
        imageio.mimsave(output_gif, gif_frames, fps=fps, loop=0)

        status = f"Masked video with green screen background saved to: {output_video}\nGIF saved to: {output_gif}\nMasks saved to: {masks_dir}\nProcessed {'test duration' if test_mode else 'full video'}"
        if progress_callback:
            progress_callback(100, status)
        return output_video, output_gif
    except Exception as e:
        error_message = f"Error in process_video: {str(e)}"
        print(error_message)
        traceback.print_exc()
        if progress_callback:
            progress_callback(-1, error_message)
        return None

# Remove the main execution block as it will now be handled by the Flask app
