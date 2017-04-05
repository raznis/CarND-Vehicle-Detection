import pickle
import numpy as np
from moviepy.video.io.VideoFileClip import VideoFileClip

from feature_extraction_utils import slide_window, search_windows, add_heat, apply_threshold, draw_labeled_bboxes
from scipy.ndimage.measurements import label

#global parameters used for both training and processing images
from processing_utils import find_cars

color_space = 'YCrCb'  # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
orient = 8  # HOG orientations
pix_per_cell = 8  # HOG pixels per cell
cell_per_block = 2  # HOG cells per block
hog_channel = "ALL"  # Can be 0, 1, 2, or "ALL"
spatial_size = (32, 32)  # Spatial binning dimensions
hist_bins = 48  # Number of histogram bins
spatial_feat = True  # Spatial features on or off
hist_feat = True  # Histogram features on or off
hog_feat = True  # HOG features on or off
window_sizes = [48, 64, 96, 128]

svc = None
X_scaler = None
_windows = []
# def process_image(image):
#     global _windows
#
#     draw_image = np.copy(image)
#     image_for_classification = image.astype(np.float32) / 255
#
#     if len(_windows) == 0:
#         for size in window_sizes:
#             _windows.extend(slide_window(image, x_start_stop=[None, None], y_start_stop=[400, 657],
#                                          xy_window=(size, size), xy_overlap=(0.7, 0.7)))
#         print("Done initializing windows.")
#
#     hot_windows = search_windows(image_for_classification, _windows, svc, X_scaler,
#                                  color_space=color_space,
#                                  spatial_size=spatial_size, hist_bins=hist_bins,
#                                  orient=orient, pix_per_cell=pix_per_cell,
#                                  cell_per_block=cell_per_block,
#                                  hog_channel=hog_channel, spatial_feat=spatial_feat,
#                                  hist_feat=hist_feat, hog_feat=hog_feat)
#
#     heat = np.zeros_like(image[:, :, 0]).astype(np.float)
#     # Add heat to each box in box list
#     heat = add_heat(heat, hot_windows)
#
#     # Apply threshold to help remove false positives
#     heat = apply_threshold(heat, 5)
#
#     # Visualize the heatmap when displaying
#     heatmap = np.clip(heat, 0, 255)
#
#     # Find final boxes from heatmap using label function
#     labels = label(heatmap)
#     draw_image = draw_labeled_bboxes(draw_image, labels)
#     return draw_image

# add global heat map
_heatmaps = []
_heatmap_sum = np.zeros((720,1280)).astype(np.float64)

ystart = 400
ystop = 657
def process_image(image):
    global _heatmaps
    global _heatmap_sum
    draw_img = np.copy(image)
    hot_windows = []
    hot_windows.extend(
        find_cars(image, ystart, ystop, 1, svc, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins,
                  color_space=color_space))
    hot_windows.extend(
        find_cars(image, ystart, ystop, 1.5, svc, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins,
                  color_space=color_space))
    hot_windows.extend(
        find_cars(image, ystart, ystop, 2, svc, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins,
                  color_space=color_space))
    hot_windows.extend(
        find_cars(image, ystart, ystop, 3, svc, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins,
                  color_space=color_space))

    heat = np.zeros_like(image[:, :, 0]).astype(np.float)
    # Add heat to each box in box list
    current_heat = add_heat(heat, hot_windows)

    _heatmap_sum += current_heat
    _heatmaps.append(current_heat)

    if len(_heatmaps) > 12:
        heat_to_remove = _heatmaps.pop(0)
        _heatmap_sum -= heat_to_remove
        _heatmap_sum = np.clip(_heatmap_sum, 0, 100000)
        heat = apply_threshold(_heatmap_sum, 25)
    else:
        # Apply threshold to help remove false positives
        heat = apply_threshold(current_heat, 3)

    # Visualize the heatmap when displaying
    heatmap = np.clip(heat, 0, 255)

    # Find final boxes from heatmap using label function
    labels = label(heatmap)
    draw_image = draw_labeled_bboxes(draw_img, labels)
    return draw_image


if __name__ == "__main__":

    svc = pickle.load(open("model/model_for_submission.sav", 'rb'))
    X_scaler = pickle.load(open('model/X_scaler.pkl', 'rb'))
    print("Loaded model and scaler from file.")
    video = 'project_video'
    white_output = '{}_test.mp4'.format(video)
    clip1 = VideoFileClip('{}.mp4'.format(video))#.subclip(35, 44)
    white_clip = clip1.fl_image(process_image)  # NOTE: this function expects color images!!
    white_clip.write_videofile(white_output, audio=False)


