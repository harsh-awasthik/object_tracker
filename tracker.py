import sys
import pyrealsense2 as rs
import cv2
import numpy as np
from function import *

# === (A) Set Up RealSense Pipeline ===
# pipe = rs.pipeline()
# cfg  = rs.config()
# cfg.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
# profile = pipe.start(cfg)
cap = cv2.VideoCapture(0)

# ---------------------------------------------------------------------------------
# NOTE: For outdoors, use the actual RealSense intrinsics. For example:
# intr = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
# CAMERA_MATRIX = np.array([[intr.fx,     0,      intr.ppx],
#                           [     0, intr.fy,     intr.ppy],
#                           [     0,     0,           1   ]], dtype=np.float32)
# DIST_COEFFS   = np.array(intr.coeffs, dtype=np.float32)
# ---------------------------------------------------------------------------------
 
# For demonstration, This is pre-calibrated Camera Matrix of Realsence and zero distortion:
CAMERA_MATRIX = np.array([
    [675.537322, 0.0,       311.191300],
    [0.0,        677.852071,221.610964],
    [0.0,        0.0,       1.0]
], dtype=np.float32)

DIST_COEFFS = np.zeros((5, 1), dtype=np.float32)  # If your cam has negligible distortion

# === (B) Define Gate Dimensions & Corresponding 3D Points ===
OBJECT_WIDTH  = 74.0 #in cms
OBJECT_HEIGHT = 74.0 #in cms
object_points  = np.array([
    [-OBJECT_WIDTH/2,  OBJECT_HEIGHT/2,  0],  # top-left
    [ OBJECT_WIDTH/2,  OBJECT_HEIGHT/2,  0],  # top-right
    [ OBJECT_WIDTH/2, -OBJECT_HEIGHT/2,  0],  # bottom-right
    [-OBJECT_WIDTH/2, -OBJECT_HEIGHT/2,  0],  # bottom-left
], dtype=np.float32)

# === (C) SIFT Detector/Matcher Setup ===
sift_detector = cv2.SIFT_create()
bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)

MIN_MATCH_COUNT = 10
win_name = "Gate Pose Estimation"

# We'll store the first reference image & corners once user selects them
img_reference = None
ref_keypoints = None
ref_descriptors = None
ref_corners_2d = None  # 4 corner points in the reference image

while True:
    # frames      = pipe.wait_for_frames()
    # color_frame = frames.get_color_frame()

    # if not color_frame:
    #     continue

    # frame = np.asanyarray(color_frame.get_data())

    ret, frame = cap.read()

    # ----------------------------------------------------------------------
    # 1. If we do not yet have a reference image (and corner selection),
    #    let user press SPACE to define the 4 corners in the current frame.
    # ----------------------------------------------------------------------
    if img_reference is None:
        # Show live feed
        text = "Press 'SPACE' to select corners"
        font = cv2.FONT_HERSHEY_COMPLEX
        (text_width, text_height), baseline = cv2.getTextSize(text, font, 0.6, 2)
        temp = frame.copy()
        cv2.putText(temp, text, (frame.shape[1]//2 - text_width//2, 20), font, 0.6, (255,0, 255), 1, cv2.LINE_AA)
        cv2.imshow(win_name, temp)
        key = cv2.waitKey(1)
        
        if key == 27:  # ESC to quit
            break
        elif key == ord(' '):
            # User selects corners on current frame
            corners = get_four_points(frame)  # (4,2) float32 array
            
            # We store a copy of the frame as the "reference" and draw the corners
            img_reference = draw_quadrilateral(frame, corners, color=(0,255,0), thickness=2)
            ref_corners_2d = np.array(corners)  # The 2D corners in the reference image

            # Also compute SIFT descriptors on the entire reference image
            gray_ref = cv2.cvtColor(img_reference, cv2.COLOR_BGR2GRAY)
            ref_keypoints, ref_descriptors = sift_detector.detectAndCompute(gray_ref, None)
        continue

    # ----------------------------------------------------------------------
    # 2. Once we have a reference image & corners, we match it against the
    #    current frame using SIFT keypoints and BFMatcher.
    # ----------------------------------------------------------------------
    gray_current = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    kp_current, desc_current = sift_detector.detectAndCompute(gray_current, None)

    if ref_descriptors is None or desc_current is None:
        cv2.imshow(win_name, frame)
        if cv2.waitKey(1) == 27:
            break
        continue

    matches = bf.match(ref_descriptors, desc_current)
    if len(matches) < MIN_MATCH_COUNT:
        # Not enough matches; just show the feed
        cv2.imshow(win_name, frame)
        if cv2.waitKey(1) == 27:
            break
        continue

    # Sort matches by distance
    good_matches = sorted(matches, key=lambda x: x.distance)

    # ----------------------------------------------------------------------
    # 3. Estimate the Homography from matched points (for robust corner warp).
    # ----------------------------------------------------------------------
    src_pts = np.float32([ref_keypoints[m.queryIdx].pt for m in good_matches])
    dst_pts = np.float32([kp_current[m.trainIdx].pt for m in good_matches])

    H, mask_h = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    if H is None:
        # Could not find a valid homography
        cv2.imshow(win_name, frame)
        if cv2.waitKey(1) == 27:
            break
        continue

    # Use homography to project the reference corners into the current image
    ref_corners_reshaped = ref_corners_2d.astype(np.float32).reshape(-1, 1, 2) # shape => (4,1,2)
    projected_corners = cv2.perspectiveTransform(ref_corners_reshaped, H)  # shape => (4,1,2)
    projected_corners = projected_corners.reshape(-1, 2)  # shape => (4,2)

    # ----------------------------------------------------------------------
    # 4. SolvePnP to get (rvec, tvec) from the known 3D corners -> image corners
    # ----------------------------------------------------------------------
    # Because the gate is planar, using SOLVEPNP_IPPE or SOLVEPNP_IPPE_SQUARE
    # can yield more stable orientation than P3P. If you prefer iterative:
    flags = cv2.SOLVEPNP_IPPE  # good for planar object
    
    success, rvec, tvec = cv2.solvePnP(
        object_points,        # shape => (4,3)
        projected_corners,    # shape => (4,2)
        CAMERA_MATRIX,
        DIST_COEFFS,
        flags=flags
    )

    # Safety check
    if not success:
        cv2.imshow(win_name, frame)
        if cv2.waitKey(1) == 27:
            break
        continue
    
    # Convert rvec to a rotation matrix
    R, _ = cv2.Rodrigues(rvec)
    
    # Convert rotation matrix to Euler angles
    yaw, pitch, roll = rotational_to_euler(R) 
    
    # (Optional) Convert from radians to degrees for easier interpretation
    yaw_deg   = np.degrees(yaw)
    pitch_deg = np.degrees(pitch)
    roll_deg  = np.degrees(roll)
    
    # ----------------------------------------------------------------------
    # 5. Visualization: draw the recognized gate corners + print pose info
    # ----------------------------------------------------------------------
    display_frame = frame.copy()
    
    # Draw the projected quadrilateral on the current frame
    cv2.polylines(display_frame, [np.int32(projected_corners)], True, (0,255,0), 3, cv2.LINE_AA)
    tvec = tvec.flatten()
    # Prepare text lines
    line1 = f"Yaw={yaw_deg:5.1f}°, Pitch={pitch_deg:5.1f}°, Roll={roll_deg:5.1f}°"
    line2 = f"Tx={tvec[0]:.2f}, Ty={tvec[1]:.2f}, Tz={tvec[2]:.2f} (cm)"
    # Note: Units of tvec depend on how you defined object_points. 
    # If object_points are in cm, then T will be in cm. 
    # If you want meters, scale them or interpret accordingly.
    
    # Overlay text
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(display_frame, line1, (10, 20), font, 0.6, (255,0, 0), 1, cv2.LINE_AA)
    cv2.putText(display_frame, line2, (10, 45), font, 0.6, (255,0, 0), 1, cv2.LINE_AA)
    
    # Show matches side-by-side as well if you like:
    match_vis = cv2.drawMatches(img_reference, ref_keypoints,
                                frame, kp_current,
                                good_matches, None,
                                matchesMask=mask_h.ravel().tolist(),
                                flags=cv2.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS)
    cv2.imshow("Matches", match_vis)
    
    cv2.imshow(win_name, display_frame)
    key = cv2.waitKey(1)
    if key == 27:  # ESC to quit
        break

cv2.destroyAllWindows()
# pipe.stop()
