import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import cv2
import glob
import time
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from skimage.feature import hog
from feature_extraction_utils import *
# NOTE: the next import is only valid for scikit-learn version <= 0.17
# for scikit-learn >= 0.18 use:
# from sklearn.model_selection import train_test_split
from sklearn.model_selection import train_test_split
from scipy.ndimage.measurements import label
import pickle
from sklearn.externals import joblib

### TODO: Tweak these parameters and see how the results change.
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


def train_model():
    # Read in cars and notcars
    non_car_images = glob.glob('data/non-vehicles/**/*.png', recursive=True)
    notcars = []
    for image in non_car_images:
        notcars.append(image)

    car_images = glob.glob('data/vehicles/**/*.png', recursive=True)
    cars = []
    for image in car_images:
        cars.append(image)

    data_info = data_look(cars, notcars)

    print('Dataset information:\n',
          data_info["n_cars"], ' cars and',
          data_info["n_notcars"], ' non-cars')
    print('of size: ', data_info["image_shape"], ' and data type:',
          data_info["data_type"])
    # # Just for fun choose random car / not-car indices and plot example images
    # car_ind = np.random.randint(0, len(cars))
    # notcar_ind = np.random.randint(0, len(notcars))
    #
    # # Read in car / not-car images
    # car_image = mpimg.imread(cars[car_ind])
    # notcar_image = mpimg.imread(notcars[notcar_ind])
    #
    # # Plot the examples
    # fig = plt.figure()
    # plt.subplot(121)
    # plt.imshow(car_image)
    # plt.title('Example Car Image')
    # plt.subplot(122)
    # plt.imshow(notcar_image)
    # plt.title('Example Not-car Image')



    car_features = extract_features(cars, color_space=color_space,
                                    spatial_size=spatial_size, hist_bins=hist_bins,
                                    orient=orient, pix_per_cell=pix_per_cell,
                                    cell_per_block=cell_per_block,
                                    hog_channel=hog_channel, spatial_feat=spatial_feat,
                                    hist_feat=hist_feat, hog_feat=hog_feat)
    notcar_features = extract_features(notcars, color_space=color_space,
                                       spatial_size=spatial_size, hist_bins=hist_bins,
                                       orient=orient, pix_per_cell=pix_per_cell,
                                       cell_per_block=cell_per_block,
                                       hog_channel=hog_channel, spatial_feat=spatial_feat,
                                       hist_feat=hist_feat, hog_feat=hog_feat)

    X = np.vstack((car_features, notcar_features)).astype(np.float64)
    # Fit a per-column scaler
    X_scaler = StandardScaler().fit(X)
    # Apply the scaler to X
    scaled_X = X_scaler.transform(X)

    #saving X_scaler to file
    scaler_filename = 'model/X_scaler.pkl'
    pickle.dump(X_scaler, open(scaler_filename, 'wb'))

    # then load it later, remember to import joblib of course


    # Define the labels vector
    y = np.hstack((np.ones(len(car_features)), np.zeros(len(notcar_features))))

    # Split up data into randomized training and test sets
    rand_state = np.random.randint(0, 100)
    X_train, X_test, y_train, y_test = train_test_split(
        scaled_X, y, test_size=0.2, random_state=rand_state)

    print('Using:', orient, 'orientations', pix_per_cell,
          'pixels per cell and', cell_per_block, 'cells per block')
    print('Feature vector length:', len(X_train[0]))
    # Use a linear SVC
    svc = LinearSVC()
    # Check the training time for the SVC
    t = time.time()
    svc.fit(X_train, y_train)
    t2 = time.time()
    print(round(t2 - t, 2), 'Seconds to train SVC...')
    # Check the score of the SVC
    accuracy = round(svc.score(X_test, y_test), 4)
    print('Test Accuracy of SVC = ', accuracy)

    # saving model to disk
    # save the model to disk
    filename = 'model/model_' + str(accuracy) + '.sav'
    pickle.dump(svc, open(filename, 'wb'))

    return svc, X_scaler

def convert_color(img, conv='RGB2YCrCb'):
    if conv == 'RGB2YCrCb':
        return cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
    if conv == 'BGR2YCrCb':
        return cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    if conv == 'RGB2LUV':
        return cv2.cvtColor(img, cv2.COLOR_RGB2LUV)


# Define a single function that can extract features using hog sub-sampling and make predictions
def find_cars(img, ystart, ystop, scale, svc, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins):
    draw_img = np.copy(img)
    img = img.astype(np.float32) / 255

    img_tosearch = img[ystart:ystop, :, :]
    ctrans_tosearch = convert_color(img_tosearch, conv='RGB2YCrCb')
    if scale != 1:
        imshape = ctrans_tosearch.shape
        ctrans_tosearch = cv2.resize(ctrans_tosearch, (np.int(imshape[1] / scale), np.int(imshape[0] / scale)))

    ch1 = ctrans_tosearch[:, :, 0]
    ch2 = ctrans_tosearch[:, :, 1]
    ch3 = ctrans_tosearch[:, :, 2]

    # Define blocks and steps as above
    nxblocks = (ch1.shape[1] // pix_per_cell) - 1
    nyblocks = (ch1.shape[0] // pix_per_cell) - 1
    nfeat_per_block = orient * cell_per_block ** 2
    # 64 was the orginal sampling rate, with 8 cells and 8 pix per cell
    window = 64
    nblocks_per_window = (window // pix_per_cell) - 1
    cells_per_step = 2  # Instead of overlap, define how many cells to step
    nxsteps = (nxblocks - nblocks_per_window) // cells_per_step
    nysteps = (nyblocks - nblocks_per_window) // cells_per_step

    # Compute individual channel HOG features for the entire image
    hog1 = get_hog_features(ch1, orient, pix_per_cell, cell_per_block, feature_vec=False)
    hog2 = get_hog_features(ch2, orient, pix_per_cell, cell_per_block, feature_vec=False)
    hog3 = get_hog_features(ch3, orient, pix_per_cell, cell_per_block, feature_vec=False)

    for xb in range(nxsteps):
        for yb in range(nysteps):
            ypos = yb * cells_per_step
            xpos = xb * cells_per_step
            # Extract HOG for this patch
            hog_feat1 = hog1[ypos:ypos + nblocks_per_window, xpos:xpos + nblocks_per_window].ravel()
            hog_feat2 = hog2[ypos:ypos + nblocks_per_window, xpos:xpos + nblocks_per_window].ravel()
            hog_feat3 = hog3[ypos:ypos + nblocks_per_window, xpos:xpos + nblocks_per_window].ravel()
            hog_features = np.hstack((hog_feat1, hog_feat2, hog_feat3))

            xleft = xpos * pix_per_cell
            ytop = ypos * pix_per_cell

            # Extract the image patch
            subimg = cv2.resize(ctrans_tosearch[ytop:ytop + window, xleft:xleft + window], (64, 64))

            # Get color features
            spatial_features = bin_spatial(subimg, size=spatial_size)
            hist_features = color_hist(subimg, nbins=hist_bins)

            # Scale features and make a prediction
            test_features = X_scaler.transform(
                np.hstack((spatial_features, hist_features, hog_features)).reshape(1, -1))
            # test_features = X_scaler.transform(np.hstack((shape_feat, hist_feat)).reshape(1, -1))
            test_prediction = svc.predict(test_features)

            if test_prediction == 1:
                xbox_left = np.int(xleft * scale)
                ytop_draw = np.int(ytop * scale)
                win_draw = np.int(window * scale)
                cv2.rectangle(draw_img, (xbox_left, ytop_draw + ystart),
                              (xbox_left + win_draw, ytop_draw + win_draw + ystart), (0, 0, 255), 6)

    return draw_img


if __name__ == "__main__":
    svc = None
    X_scaler = None

    #svc, X_scaler = train_model()

    if svc is None or X_scaler is None:
        loaded_model = pickle.load(open("model/model_0.9899.sav", 'rb'))
        loaded_X_scaler = pickle.load(open('model/X_scaler.pkl', 'rb'))
        print("Loaded model from file.")
    else:
        loaded_model = svc
        loaded_X_scaler = X_scaler
        print("Used model from training.")


    ystart = 400
    ystop = 750
    scale = 2


    # out_img = find_cars(draw_image, ystart, ystop, scale, loaded_model, loaded_X_scaler, orient, pix_per_cell, cell_per_block, spatial_size,
    #                     hist_bins)

    image = mpimg.imread('test_images/test6.jpg')
    draw_image = np.copy(image)
    image_for_classification = image.astype(np.float32) / 255

    window_sizes = [48, 64, 96, 128]
    windows = []
    for size in window_sizes:
        windows.extend(slide_window(image_for_classification, x_start_stop=[None, None], y_start_stop=[400,657],
                                                  xy_window=(size, size), xy_overlap=(0.7, 0.7)))


    hot_windows = search_windows(image_for_classification, windows, loaded_model, loaded_X_scaler, color_space=color_space,
                                 spatial_size=spatial_size, hist_bins=hist_bins,
                                 orient=orient, pix_per_cell=pix_per_cell,
                                 cell_per_block=cell_per_block,
                                 hog_channel=hog_channel, spatial_feat=spatial_feat,
                                 hist_feat=hist_feat, hog_feat=hog_feat)


    heat = np.zeros_like(image[:, :, 0]).astype(np.float)
    # Add heat to each box in box list
    heat = add_heat(heat, hot_windows)

    # Apply threshold to help remove false positives
    heat = apply_threshold(heat, 2)

    # Visualize the heatmap when displaying
    heatmap = np.clip(heat, 0, 255)

    # Find final boxes from heatmap using label function
    labels = label(heatmap)
    draw_img = draw_labeled_bboxes(np.copy(image), labels)

    # window_img = draw_boxes(draw_image, hot_windows, color=(0, 0, 255), thick=2)
    # cv2.imwrite("output_images/window_test5.jpg", cv2.cvtColor(draw_img, cv2.COLOR_RGB2BGR))
    cv2.imwrite("output_images/window_test6.jpg", cv2.cvtColor(draw_img,cv2.COLOR_RGB2BGR))




    # y_start_stop = [400, 700]  # Min and max in y to search in slide_window()
    # # Check the prediction time for a single sample
    # t = time.time()

    # Uncomment the following line if you extracted training
    # data from .png images (scaled 0 to 1 by mpimg) and the
    # image you are searching is a .jpg (scaled 0 to 255)
    # image = image.astype(np.float32)/255
    #
    # windows = slide_window(image, x_start_stop=[None, None], y_start_stop=y_start_stop,
    #                        xy_window=(80, 80), xy_overlap=(0.5, 0.5))
    #
    # hot_windows = search_windows(image, windows, loaded_model, loaded_X_scaler, color_space=color_space,
    #                              spatial_size=spatial_size, hist_bins=hist_bins,
    #                              orient=orient, pix_per_cell=pix_per_cell,
    #                              cell_per_block=cell_per_block,
    #                              hog_channel=hog_channel, spatial_feat=spatial_feat,
    #                              hist_feat=hist_feat, hog_feat=hog_feat)
    #
    # window_img = draw_boxes(draw_image, hot_windows, color=(0, 0, 255), thick=6)
    #
    # # plt.imshow(window_img)
    # window_img = cv2.cvtColor(window_img, cv2.COLOR_RGB2BGR)
    # cv2.imwrite("output_images/window_img_from_training.jpg", window_img)






