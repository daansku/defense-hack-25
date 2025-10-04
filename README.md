Here's a list of all the functions in the code and what they do:

Core Motion Detection Functions

1. detect_motion(image1, image2, thresh, save_crop, original_frame, output_dir)
   - Compares two grayscale images to detect motion
   - Creates a binary map where pixels above threshold are marked as motion
   - Applies morphological operations to clean up noise
   - Finds bounding box of motion area
   - Optionally saves cropped image of detected motion
   - Returns: (image2 if motion detected, number of changed pixels, bounding box, difference visualization)

2. find_bounding_box(diff_map, min_size)
   - Takes a binary motion map and finds the smallest rectangle that contains all motion pixels
   - Adds 10 pixels of padding around the motion area
   - Filters out boxes that are too small (< min_size)
   - Filters out boxes that cover >90% of image (likely false positives)
   - Returns: (x1, y1, x2, y2) coordinates or None

Image Processing Functions

3. convert_to_greyscale(image_path)
   - Loads an image file and converts it to grayscale
   - Normalizes pixel values to 0-1 range
   - Returns: numpy array of shape (H, W) with values [0, 1]

4. crop_and_compress(frame, bbox, quality)
   - Crops a frame to the specified bounding box
   - Applies JPEG compression to reduce file size
   - Validates bbox coordinates are within bounds
   - Returns: compressed cropped image or None if error

## File Management Functions

5. save_frame(frame, bbox, output_dir, is_visualization)
   - Saves a frame to disk with timestamp
   - Optionally crops and compresses before saving
   - Adds prefix "viz_" for visualization images, "capture_" for regular captures
   - Returns: filename of saved image

6. delete_oldest_image(folder, keep_count)
   - Keeps only the N most recent images in a folder
   - Deletes older images when count exceeds keep_count
   - Useful for managing disk space

7. delete_all_images(folder_path)
   - Deletes all image files (.jpg, .jpeg, .png, .bmp) in a folder
   - Useful for clearing out folders on startup
   - Logs how many images were deleted

Batch Processing Function

8. process_folder_images(folder_path)
   - Processes existing images in a folder
   - Uses oldest image as background reference
   - Uses newest image to detect motion
   - Runs motion detection and saves results
   - Returns: (background_gray, newest_frame, newest_gray, bbox) or None

Main Program

9. main()
   - Entry point for the program
   - Creates necessary directories
   - Processes any existing images in folder
   - Opens camera and starts live motion detection
   - Processes every Nth frame (defined by FRAME_SKIP)
   - Displays live video feed and difference visualization
   - Exits when 'q' key is pressed

Constants at Top of File

- PIXEL_DIFF_THRESH = 0.1 - Threshold for detecting motion (0-1 range, 0.1 = 10% difference)
- FRAME_SKIP = 60 - Process every 60th frame to reduce computational load
- CAMERA_ID = 1 - Which camera to use (0 = built-in webcam, 1 = external camera)