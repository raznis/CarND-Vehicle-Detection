import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import cv2
import glob
import time
import pickle
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from feature_extraction_utils import data_look, extract_features
from processing_utils import find_cars_in_image

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


if __name__ == "__main__":
    svc = None
    X_scaler = None

    #svc, X_scaler = train_model()

    if svc is None or X_scaler is None:
        loaded_model = pickle.load(open("model/model_for_submission.sav", 'rb'))
        loaded_X_scaler = pickle.load(open('model/X_scaler.pkl', 'rb'))
        print("Loaded model from file.")
    else:
        loaded_model = svc
        loaded_X_scaler = X_scaler
        print("Used model from training.")


    # ystart = 400
    # ystop = 750
    # scale = 2
    #

    # out_img = find_cars(draw_image, ystart, ystop, scale, loaded_model, loaded_X_scaler, orient, pix_per_cell, cell_per_block, spatial_size,
    #                     hist_bins)
    for i in range(1,7):
        image = mpimg.imread('test_images/test' + str(i) + '.jpg')
        # image_with_cars = process_image(image, loaded_model, loaded_X_scaler)
        image_with_cars = find_cars_in_image(image, 400, 657, 1, loaded_model, loaded_X_scaler, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins, color_space)
        cv2.imwrite('output_images/window_test' + str(i) + '.jpg', cv2.cvtColor(image_with_cars,cv2.COLOR_RGB2BGR))









