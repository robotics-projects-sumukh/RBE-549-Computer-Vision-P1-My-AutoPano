import cv2
import numpy as np
from scipy.ndimage import maximum_filter
import os
from tqdm import tqdm

def create_corners_overlay(image, corner_coords, color=(0, 0, 255)):
    overlay = image.copy()

    for row, col in corner_coords:  # Swapping to (col, row) format
        cv2.circle(overlay, (int(col), int(row)), 1, color, -1)

    return overlay

def drawImage(image, name, set, set_type, image_type, index):
    if not os.path.exists(f"Results/{set_type}/{set}/{image_type}"):
        os.makedirs(f"Results/{set_type}/{set}/{image_type}")
        
    cv2.imwrite(f"Results/{set_type}/{set}/{image_type}/{name}_{index}.png", image)

# Convert the anms_points to cv2.KeyPoint objects
def convert_to_keypoints(coords):
    return [cv2.KeyPoint(x=float(col), y=float(row), size=1) for row, col in coords]

# Convert the matches to DMatch objects
def convert_to_dmatches(matches):
    return [cv2.DMatch(_queryIdx=match[0], _trainIdx=match[1], _distance=0) for match in matches]

def get_corner(img_orig, blockSize=2, ksize=9, k=0.22):
    corner_threshold = 0.0001
    img = img_orig.copy()
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_img = np.float32(gray_img)
    dst = cv2.cornerHarris(gray_img, blockSize, ksize, k)
    # Check for need of dilation or not
    # dst = cv2.dilate(dst, None)
    # Threshold for an optimal value, it may vary depending on the image.
    c_pts = np.where(dst > corner_threshold * dst.max())
    dst[dst <= corner_threshold * dst.max()] = 0
    c_scores = dst
    img[dst > corner_threshold * dst.max()] = [0, 0, 255]
    c_img = img
    return c_pts, c_scores, c_img

def ANMS(Cimg, NumFeatures):
    """
    Adaptive Non-Maximal Suppression (optimized version)
    """
    # Find local maxima
    lm = maximum_filter(Cimg, size=5) == Cimg
    threshold = 0.0000001 * Cimg.max()
    msk = (lm & (Cimg > threshold))
    
    # Get coordinates and values of local maxima
    lm_coords = np.argwhere(msk)
    Nstrong = len(lm_coords)

    if Nstrong == 0:
        raise ValueError("No local maxima found.")
    
    # Get values at local maxima
    lm_values = Cimg[tuple(lm_coords.T)]
    
    # Vectorized computation of distances
    # Reshape coordinates for broadcasting
    coords1 = lm_coords[:, np.newaxis, :]  # Shape: (N, 1, 2)
    coords2 = lm_coords[np.newaxis, :, :]  # Shape: (1, N, 2)
    
    # Compute all pairwise distances at once
    distances = np.sqrt(np.sum((coords1 - coords2) ** 2, axis=2))  # Shape: (N, N)
    
    # Create mask for valid comparisons
    value_mask = lm_values[:, np.newaxis] < lm_values[np.newaxis, :]  # Shape: (N, N)
    
    # Set invalid distances to infinity
    distances[~value_mask] = np.inf
    distances[np.eye(Nstrong, dtype=bool)] = np.inf  # Exclude self-distances
    
    # Get minimum distance for each point
    radius = np.min(distances, axis=1)
    
    # Sort indices based on radius
    sorted_indices = np.argsort(radius)[::-1]
    
    # Filter points within bounds more efficiently
    valid_mask = ((lm_coords[sorted_indices, 0] >= 21) & 
                 (lm_coords[sorted_indices, 0] < Cimg.shape[0] - 21) &
                 (lm_coords[sorted_indices, 1] >= 21) & 
                 (lm_coords[sorted_indices, 1] < Cimg.shape[1] - 21))
    
    # Get final points
    valid_indices = sorted_indices[valid_mask]
    NumFeatures = min(NumFeatures, len(valid_indices))
    anms_points = lm_coords[valid_indices[:NumFeatures]]
    
    return anms_points.tolist()  # Convert to list for compatibility

def get_feature_descriptor(anms_points, Image):
    feature_descriptor = []
    coordinates = []
    for j in range(len(anms_points)):
        row, col = anms_points[j]
        row, col = int(row), int(col)
        # Boundary check
        if row - 20 < 0 or row + 21 > Image.shape[0] or col - 20 < 0 or col + 21 > Image.shape[1]:
            # print(f"Skipping point ({row}, {col}) near the border.")
            continue  # Skip points near the border
        
        patch = Image[row-20:row+21, col-20:col+21]
        # Apply Gaussian blur
        patch = cv2.GaussianBlur(patch, (5, 5), 0)
        # Sub-sample and standardize
        patch = cv2.resize(patch, (8, 8))
        patch = patch.flatten()
        patch = (patch - np.mean(patch)) / np.std(patch)
        feature_descriptor.append(patch)
        coordinates.append((row, col))
    # print ("Number of feature descriptors: ", len(feature_descriptor))
    return feature_descriptor, coordinates

def match_features(feature_descriptor1, feature_descriptor2, ratio_threshold=0.75):
    matches = []
    for j in range(len(feature_descriptor1)):
        best_match = -1
        best_match_dist = np.inf
        second_best_match_dist = np.inf
        for k in range(len(feature_descriptor2)):
            dist = np.linalg.norm(feature_descriptor1[j] - feature_descriptor2[k])
            if dist < best_match_dist:
                second_best_match_dist = best_match_dist
                best_match_dist = dist
                best_match = k
        if best_match_dist / second_best_match_dist < ratio_threshold:
            matches.append((j, best_match))
            # matches.append((feature_descriptor1[j], feature_descriptor2[best_match]))
    return matches

def find_best_matching_pairs(images, blockSize=2, ksize=3, k=0.03, nBest=1000):
    """
    Find the best matching pairs of images based on the number of good feature matches.
    Returns a list of tuples (i, j, num_matches) sorted by number of matches.
    """
    image_pairs = []
    for i in range(len(images)):
        for j in range(i + 1, len(images)):
            # print(f"Matching images {i+1} and {j+1}...")
            # Get corners and ANMS points for both images
            _, corner_score1, _ = get_corner(images[i], blockSize=blockSize, ksize=ksize, k=k)
            _, corner_score2, _ = get_corner(images[j], blockSize=blockSize, ksize=ksize, k=k)
            
            anms_points1 = ANMS(corner_score1, nBest)
            anms_points2 = ANMS(corner_score2, nBest)
            
            # Get feature descriptors
            feature_descriptor1, _ = get_feature_descriptor(anms_points1, images[i])
            feature_descriptor2, _ = get_feature_descriptor(anms_points2, images[j])
            
            # Match features
            matches = match_features(feature_descriptor1, feature_descriptor2)
            
            # Store the pair and number of matches
            image_pairs.append((i, j, len(matches)))
    
    # Sort pairs by number of matches in descending order
    return sorted(image_pairs, key=lambda x: x[2], reverse=True)

def create_panorama_graph(image_pairs, num_images):
    """
    Create a graph representation of image connections
    Returns adjacency list and weights dictionary
    """
    graph = {i: [] for i in range(num_images)}
    weights = {}
    
    for i, j, matches in image_pairs:
        graph[i].append(j)
        graph[j].append(i)
        weights[(i, j)] = weights[(j, i)] = matches
    
    return graph, weights

def find_optimal_ordering(graph, weights, start_node):
    """
    Find optimal ordering of images using a modified DFS
    Returns ordered list of image indices
    """
    visited = set()
    ordered = []
    
    def dfs(node):
        if node in visited:
            return
        visited.add(node)
        ordered.append(node)
        
        # Sort neighbors by weight
        neighbors = [(n, weights.get((node, n), 0)) for n in graph[node]]
        neighbors.sort(key=lambda x: x[1], reverse=True)
        
        for neighbor, _ in neighbors:
            if neighbor not in visited:
                dfs(neighbor)
    
    dfs(start_node)
    return ordered

def crop_black_edges(panorama, threshold=10):
    """
    Removes excess black (or near-black) regions from the panorama image.
    
    Args:
        panorama: Input panorama image
        threshold: Pixel intensity threshold to consider as "black"
        
    Returns:
        Cropped panorama with black edges removed
    """
    if panorama is None:
        return None
        
    # Convert to grayscale
    if len(panorama.shape) == 3:
        gray = cv2.cvtColor(panorama, cv2.COLOR_BGR2GRAY)
    else:
        gray = panorama.copy()
    
    # Create a binary mask of non-black pixels
    _, mask = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
    
    # Find contours of non-black regions
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return panorama
        
    # Find the largest contour (main panorama content)
    largest_contour = max(contours, key=cv2.contourArea)
    
    # Get bounding rectangle
    x, y, w, h = cv2.boundingRect(largest_contour)
    
    # Add a small padding (optional)
    padding = 5
    x = max(0, x - padding)
    y = max(0, y - padding)
    w = min(panorama.shape[1] - x, w + 2 * padding)
    h = min(panorama.shape[0] - y, h + 2 * padding)
    
    # Crop the image
    cropped = panorama[y:y+h, x:x+w]
    
    return cropped

def ransac_homography(matches):
    inlier_percent_threshold = 90
    num_iterations = 5000
    threshold = 10
    best_inliers = []
    best_H = None
    num_matches = len(matches)
    required_inliers = int(inlier_percent_threshold / 100 * num_matches)
    new_matches = []
    for match in matches:
        new_matches.append((match[0][::-1], match[1][::-1]))
    matches = new_matches
    for _ in range(num_iterations):
        random_indices = np.random.choice(num_matches, 4, replace=False)
        src_keypoints = np.float32([matches[i][0] for i in random_indices]).reshape(
            -1, 1, 2
        )
        dst_keypoints = np.float32([matches[i][1] for i in random_indices]).reshape(
            -1, 1, 2
        )
        H = cv2.getPerspectiveTransform(
            src_keypoints.squeeze(), dst_keypoints.squeeze()
        )
        # Compute Inliers
        inliers = []
        for match in matches:
            temp_src = np.float32(match[0]).reshape(-1, 1, 2)
            temp_dst = np.float32(match[1]).reshape(-1, 1, 2)
            transformed_points = cv2.perspectiveTransform(temp_src, H)
            ssd = np.sum((transformed_points - temp_dst) ** 2)
            if ssd < threshold:
                inliers.append(match)

        # Keep the largest set of inliers
        if len(inliers) > len(best_inliers):
            best_inliers = inliers
            best_H = H

        # Check the stopping condition
        if len(best_inliers) >= required_inliers:
            # print("Required number of inliers found!")
            break

    # Compute Inliers
    # print("Number of Inliers: ", len(best_inliers))

    # Recompute Homography using all the inliers
    if best_inliers:
        src_keypoints = np.float32([match[0] for match in best_inliers]).reshape(-1, 1, 2)
        dst_keypoints = np.float32([match[1] for match in best_inliers]).reshape(-1, 1, 2)
        if len(src_keypoints) < 4:
            # print("Not enough inliers found!")
            return None, None
        best_H, _ = cv2.findHomography(src_keypoints, dst_keypoints, method=cv2.RANSAC, ransacReprojThreshold=threshold)
    else:
        # print("No inliers found!")
        best_H = None
    return best_H, best_inliers

def check_duplicate_matches(inliers):
    """
    Check if there are duplicate source or destination points in the inliers
    Returns True if duplicates found, False otherwise
    """
    if not inliers:
        return True
        
    # Convert points to tuples for hashability
    src_points = [tuple(match[0]) for match in inliers]
    dst_points = [tuple(match[1]) for match in inliers]

    # Check for duplicates using sets
    if len(src_points) - len(set(dst_points)) > 0.1*len(src_points) or len(dst_points) - len(set(src_points)) > 0.1*len(dst_points):
        return True
        
    return False

def plot_ransac_inliers(img1, img2, keypoints_1, keypoints_2, inliers):
    # Convert inliers to DMatch objects
    dmatches = []
    for idx, (src_pt, dst_pt) in enumerate(inliers):
        # Find the indices of the keypoints that match the inlier points
        src_idx = next(i for i, kp in enumerate(keypoints_1) 
                      if abs(kp.pt[0] - src_pt[0]) < 1e-5 and 
                         abs(kp.pt[1] - src_pt[1]) < 1e-5)
        dst_idx = next(i for i, kp in enumerate(keypoints_2) 
                      if abs(kp.pt[0] - dst_pt[0]) < 1e-5 and 
                         abs(kp.pt[1] - dst_pt[1]) < 1e-5)
        dmatches.append(cv2.DMatch(src_idx, dst_idx, 0))

    # Create the visualization
    inlier_image = cv2.drawMatches(
        img1, keypoints_1, 
        img2, keypoints_2, 
        dmatches, 
        None,
        matchColor=(0, 255, 0),      # Green color for matches
        singlePointColor=(255, 0, 0), # Red color for keypoints
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    )
    return inlier_image

def create_weighted_mask(mask):
    # Ensure the mask is single-channel and 8-bit
    if len(mask.shape) > 2:
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    if mask.dtype != np.uint8:
        mask = mask.astype(np.uint8)
    # Create a distance transform
    distance = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
    # Normalize to get weights between 0 and 1
    weighted_mask = cv2.normalize(distance, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
    return weighted_mask

def get_blended_warped_image(img1, img2, best_H):
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]

    # Warp img2 to img1's plane
    corners_img2 = np.float32([[0, 0], [w2, 0], [w2, h2], [0, h2]]).reshape(-1, 1, 2)
    warped_corners_img2 = cv2.perspectiveTransform(corners_img2, best_H)

    # Combine corners to get bounds for the panorama
    corners_img1 = np.float32([[0, 0], [w1, 0], [w1, h1], [0, h1]]).reshape(-1, 1, 2)
    all_corners = np.vstack((corners_img1, warped_corners_img2))
    [xmin, ymin] = np.int32(all_corners.min(axis=0).ravel())
    [xmax, ymax] = np.int32(all_corners.max(axis=0).ravel())

    # Translation matrix
    translate = np.array([[1, 0, -xmin], [0, 1, -ymin], [0, 0, 1]])

    # Warp img2
    img2_warped = cv2.warpPerspective(img2, translate.dot(best_H), (xmax - xmin, ymax - ymin))

    # Translate img1
    img1_translated = np.zeros_like(img2_warped)
    img1_translated[-ymin:h1 - ymin, -xmin:w1 - xmin] = img1

    # Create masks for blending
    # mask1 = (img1_translated > 0).astype(np.uint8) * 255
    # mask2 = (img2_warped > 0).astype(np.uint8) * 255

    # Create masks for blending
    mask1 = cv2.cvtColor((img1_translated > 0).astype(np.uint8) * 255, cv2.COLOR_BGR2GRAY)
    mask2 = cv2.cvtColor((img2_warped > 0).astype(np.uint8) * 255, cv2.COLOR_BGR2GRAY)


    # Generate weighted masks
    weighted_mask1 = create_weighted_mask(mask1)
    weighted_mask2 = create_weighted_mask(mask2)

    # Combine the images using weighted masks
    result = (img1_translated * weighted_mask1[..., None] + img2_warped * weighted_mask2[..., None]) / \
             (weighted_mask1[..., None] + weighted_mask2[..., None] + 1e-10)  # Add a small value to avoid division by zero

    return np.uint8(result)

def get_warped_image(img1, img2, best_H):
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]

    # Warp img2 to img1's plane
    corners_img2 = np.float32([[0, 0], [w2, 0], [w2, h2], [0, h2]]).reshape(-1, 1, 2)
    warped_corners_img2 = cv2.perspectiveTransform(corners_img2, best_H)

    # Combine corners to get bounds for the panorama
    corners_img1 = np.float32([[0, 0], [w1, 0], [w1, h1], [0, h1]]).reshape(-1, 1, 2)
    all_corners = np.vstack((corners_img1, warped_corners_img2))
    [xmin, ymin] = np.int32(all_corners.min(axis=0).ravel())
    [xmax, ymax] = np.int32(all_corners.max(axis=0).ravel())

    # Translation matrix
    translate = np.array([[1, 0, -xmin], [0, 1, -ymin], [0, 0, 1]], dtype=np.float32)

    # Warp img2
    result = cv2.warpPerspective(img2, translate.dot(best_H), (xmax - xmin, ymax - ymin))
    result[-ymin:h1 - ymin, -xmin:w1 - xmin] = img1    

    return result

def get_panorama(img1, img2, blockSize=2, ksize=3, k=0.03, nBest=1000):
    flag = False
    """
    Corner Detection
    Save Corner detection output as corners.png
    """
    corner_points1, corner_score1, Cimg1 = get_corner(img1, blockSize=blockSize, ksize=ksize, k=k)
    corner_points2, corner_score2, Cimg2 = get_corner(img2, blockSize=blockSize, ksize=ksize, k=k)
    # print("Corner detection completed.")
    
    corner_coords1 = np.argwhere(corner_score1 > 0.001 * corner_score1.max())
    corner_coords2 = np.argwhere(corner_score2 > 0.001 * corner_score2.max())
    corner_overlay1 = create_corners_overlay(img1, corner_coords1, color=(0, 0, 255))
    corner_overlay2 = create_corners_overlay(img2, corner_coords2, color=(0, 0, 255))
    """
    Perform ANMS: Adaptive Non-Maximal Suppression
    Save ANMS output as anms.png
    """
    anms_points1 = ANMS(corner_score1, nBest)
    anms_points2 = ANMS(corner_score2, nBest)
    # print("ANMS completed.")
    
    anms_overlay1 = create_corners_overlay(img1, anms_points1, color=(0, 0, 255))
    anms_overlay2 = create_corners_overlay(img2, anms_points2, color=(0, 0, 255))
    """
    Feature Descriptors
    Save Feature Descriptor output as FD.png
    """
    feature_descriptor1, coordinates_1 = get_feature_descriptor(anms_points1, img1)
    feature_descriptor2, coordinates_2 = get_feature_descriptor(anms_points2, img2)
    # print("Feature descriptors computed.")
    
    fd_img_1 = img1.copy()
    for cord in coordinates_1:
        x1, y1 = cord
        cv2.circle(fd_img_1, (y1, x1), 2, (255,0,0), -1)
    fd_img_2 = img2.copy()
    for cord in coordinates_2:
        x2, y2 = cord
        cv2.circle(fd_img_2, (y2,x2), 2, (255,0,0), -1)
    
    """
    Feature Matching
    Save Feature Matching output as matching.png
    """
    keypoints_1 = convert_to_keypoints(anms_points1)
    keypoints_2 = convert_to_keypoints(anms_points2)
    matches = match_features(feature_descriptor1, feature_descriptor2)
    # print(f"Number of matches found: {len(matches)}")
    dmatches = convert_to_dmatches(matches)
    matching_image = cv2.drawMatches(img1, keypoints_1, img2, keypoints_2, dmatches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    """
    Refine: RANSAC, Estimate Homography
    """
    keypoints = [(anms_points1[i], anms_points2[j]) for i, j in matches]
    if len(keypoints) < 4:
        # print("Not enough matches found!")
        flag = True
        return None, flag
    H, inliers = ransac_homography(keypoints)

    # Check for duplicate matches in inliers
    if H is None or check_duplicate_matches(inliers):
        # print("Duplicate matches found or no homography computed!")
        flag = True
        return None, flag
    
    
    return H, flag