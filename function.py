import cv2
import numpy as np

def rotational_to_euler(R):
    """
    Converts a 3x3 rotation matrix R into Euler angles (yaw, pitch, roll)
    in the Z-Y-X convention (common in robotics/vision).
    
    Returns angles in radians.
    
    NOTE: Gimbal lock near pitch = ±90° can cause ambiguous yaw/roll.
    """
    # Handle gimbal lock scenarios
    if np.isclose(R[2, 0], -1.0):
        # pitch = +90 deg 
        pitch = np.pi / 2
        yaw = np.arctan2(R[0, 1], R[0, 2])
        roll = 0.0
    elif np.isclose(R[2, 0], 1.0):
        # pitch = -90 deg
        pitch = -np.pi / 2
        yaw = np.arctan2(-R[0, 1], -R[0, 2])
        roll = 0.0
    else:
        # General case
        pitch = -np.arcsin(R[2, 0])  # around Y
        roll  =  np.arctan2(R[2, 1]/np.cos(pitch), R[2, 2]/np.cos(pitch))  # around X
        yaw   =  np.arctan2(R[1, 0]/np.cos(pitch), R[0, 0]/np.cos(pitch))  # around Z

    return yaw, pitch, roll

def draw_quadrilateral(image, pts, color=(0,255,0), thickness=2):
    """
    Draws a quadrilateral given by pts (4 corner points) on the image.
    pts should be a list/array of shape (4,2).
    """
    pts = np.array(pts, dtype=np.int32)
    
    # Create a blank mask of the same size as the image
    mask = np.zeros_like(image)
    
    # Fill the quadrilateral area on the mask
    cv2.fillPoly(mask, [pts], (255, 255, 255))
    
    # Apply the mask to the original image
    masked_image = cv2.bitwise_and(image, mask)
    
    # Draw the quadrilateral frame on the image
    cv2.polylines(masked_image, [pts], isClosed=True, color=(0, 255, 255), thickness=3)
    
    return masked_image

def get_four_points(image):
    text = "Select 4 Corners"
    text2 = "Top Left - Top Right - Bottom Right - Bottom Left"
    font = cv2.FONT_HERSHEY_COMPLEX
    (text_width, text_height), baseline = cv2.getTextSize(text, font, 0.6, 1)
    (text2_width, text_height), baseline = cv2.getTextSize(text2, font, 0.6, 1)
    temp_image = image.copy()
    cv2.putText(temp_image, text, (image.shape[1]//2 - text_width//2, 20), font, 0.6, (255,0, 255), 1, cv2.LINE_AA)
    cv2.putText(temp_image, text2, (image.shape[1]//2 - text2_width//2, 45), font, 0.6, (255,0, 255), 1, cv2.LINE_AA)
    points = []

    def mouse_callback(event, x, y, flags, param):
        # Record a point on left mouse button click
        if event == cv2.EVENT_LBUTTONDOWN:
            points.append((x, y))
            print(f"Point {len(points)} selected: ({x}, {y})")
            # Draw the point on the image
            cv2.circle(temp_image, (x, y), 5, (0, 255, 0), -1)
            cv2.imshow("Select 4 Points", temp_image)
    
    cv2.imshow("Select 4 Points", temp_image)
    cv2.setMouseCallback("Select 4 Points", mouse_callback)

    # Wait until 4 points are selected
    while len(points) < 4:
        cv2.waitKey(1)

    cv2.destroyAllWindows()
    return points

