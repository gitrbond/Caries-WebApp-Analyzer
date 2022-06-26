import os
import pathlib
import numpy as np
import os
import tensorflow as tf
import ntpath

from PIL import Image
from IPython.display import display
# from object_detection import utils
from object_detection.utils import ops as utils_ops
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util
from datetime import datetime
from pathlib import Path

# patch tf1 into `utils.ops`
utils_ops.tf = tf.compat.v1
# Patch the location of gfile
tf.gfile = tf.io.gfile

PATH_TO_LABELS = 'cav_detection_tf_obj_det_api/my_custom_detector/training/object-detection.pbtxt'
category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS, use_display_name=True)

# If you want to test the code with your images, just add path to the images to the TEST_IMAGE_PATHS.
PATH_TO_TEST_IMAGES_DIR = pathlib.Path('images_test')
PATH_TO_SAVE_DETECTED = pathlib.Path('images_analyzed')
PATH_TO_SAVE_RAW = 'images_raw'
TEST_IMAGE_PATHS = sorted(list(PATH_TO_TEST_IMAGES_DIR.glob("*.jpg")))
print(TEST_IMAGE_PATHS)


def load_model():
    # model_dir = "/content/drive/My Drive/cav_detection_tf_obj_det_api/my_custom_detector/trained-inference-graphs/saved_model"
    model_dir = "cav_detection_tf_obj_det_api/my_custom_detector/trained-inference-graphs/saved_model"
    model = tf.compat.v2.saved_model.load(model_dir, None)
    model = model.signatures['serving_default']
    return model


detection_model = load_model()

tf.compat.v1.disable_eager_execution()  # very important line of code

from flask import Flask, request, render_template, send_from_directory
from datetime import datetime
from pathlib import Path

app = Flask(__name__)
APP_ROOT = os.path.dirname(os.path.abspath(__file__))


@app.route('/')
@app.route('/upload_form')
def my_index():
    return render_template("uploading.html")


@app.route("/file_upload", methods=["POST"])
def file_upload():
    target = os.path.join(APP_ROOT, PATH_TO_SAVE_RAW)
    # print(target)
    if not os.path.isdir(target):
        os.mkdir(target)
    print(request.files.getlist("file"))
    # global count
    # flag = 0
    for upload in request.files.getlist("file"):
        print(upload)
        full_filename = upload.filename #with extension
        print("{} is the file name".format(full_filename))
        # This is to verify files are supported
        ext = os.path.splitext(full_filename)[1]
        filename = Path(full_filename).stem
        if (ext.lower() == ".jpg") or (ext.lower() == ".png") or (ext.lower() == ".bmp"):
            print("Ok, file supported")
        else:
            print(f"file {full_filename} not supported")
            flag = 1
            return render_template("fileformat_error.html")
            # break
        now = datetime.now()
        date_time_str = now.strftime("%Y%m%d-%H%M%S")
        save_destination = target + "/" + filename + "_" + date_time_str + ext
        print("Accept incoming file:", filename)
        print("Save it to:", save_destination)
        upload.save(save_destination)
        name = target + '/' + str(upload.filename)
        is_caries_found, image_analyzed_fullpath = save_inference(detection_model, save_destination)
        return render_template("results.html", image_raw_fullpath=save_destination, image_analyzed_fullpath=image_analyzed_fullpath, detected=is_caries_found)

@app.route('/file_upload/<file_path>', methods=['GET','POST'])
def send_image(file_path):
    print("SENDING IMAGE")
    full_filename = ntpath.basename(file_path) # with ext, Linux compatible
    print(full_filename)
    return send_from_directory(PATH_TO_SAVE_RAW, full_filename)
"""
@app.route('/upload/<file_path>')
def send_image_analyzed(file_path):
    full_filename = ntpath.basename(file_path)  # with ext, Linux compatible
    print(full_filename)
    return send_from_directory(PATH_TO_SAVE_DETECTED, full_filename)
"""
#@app.route('/upload/<string:filepath>')
#def upload_from_filepath(filepath):
#    return f"Upload file '{filepath}'"


def run_inference_for_single_image(model, image):
    image = np.asarray(image)
    # The input needs to be a tensor, convert it using `tf.convert_to_tensor`.
    input_tensor = tf.convert_to_tensor(image)
    # The model expects a batch of images, so add an axis with `tf.newaxis`.
    input_tensor = input_tensor[tf.newaxis, ...]

    # Run inference
    output_dict = model(input_tensor)

    # All outputs are batches tensors.
    # Convert to numpy arrays, and take index [0] to remove the batch dimension.
    # We're only interested in the first num_detections.
    with tf.compat.v1.Session() as sess:
        #print(output_dict['num_detections'].eval())  # debugging line
        num_detections = int(output_dict.pop('num_detections').eval()[0])
        print(f"num detections={num_detections}")

        output_dict = {key: value[0, :num_detections].eval() for key, value in output_dict.items()}
        output_dict['num_detections'] = num_detections

        # detection_classes should be ints.
        output_dict['detection_classes'] = output_dict['detection_classes'].astype(np.int64)

    # Handle models with masks:
    if 'detection_masks' in output_dict:
        # Reframe the the bbox mask to the image size.
        detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
            output_dict['detection_masks'], output_dict['detection_boxes'],
            image.shape[0], image.shape[1])
        detection_masks_reframed = tf.cast(detection_masks_reframed > 0.4,
                                           tf.uint8)
        output_dict['detection_masks_reframed'] = detection_masks_reframed.numpy()

    return output_dict, num_detections


def save_inference(model, image_path):
    filename = Path(image_path).stem
    # the array based representation of the image will be used later in order to prepare the
    # result image with boxes and labels on it.
    print("debug " + image_path)
    image_np = np.array(Image.open(image_path))
    # Actual detection.
    output_dict, num_detections = run_inference_for_single_image(model, image_np)
    #print(output_dict['num_detections'])
    # print(type(category_index))
    #print(category_index)
    category_index.update({1: {'id': 1, 'name': 'kapuec'}})
    # print (category_index)
    # Visualization of the results of a detection.
    vis_util.visualize_boxes_and_labels_on_image_array(
        image_np,
        output_dict['detection_boxes'],
        output_dict['detection_classes'],
        output_dict['detection_scores'],
        category_index,
        instance_masks=output_dict.get('detection_masks_reframed', None),
        use_normalized_coordinates=True,
        line_thickness=6)

    # display(Image.fromarray(image_np))
    # cnt = 0
    im = Image.fromarray(image_np)
    now = datetime.now()
    date_time_str = now.strftime("%Y%m%d-%H%M%S")
    # print('DateTime String:', date_time_str)
    path_to_save = f"{PATH_TO_SAVE_DETECTED}/test1"
    if not os.path.isdir(path_to_save):
        # os.mkdirs(path_to_save)
        os.makedirs(path_to_save, exist_ok=True)
    image_analyzed_fullpath = path_to_save + "/" + filename + "_analyzed_" + date_time_str + ".jpg"
    #im.save(f"{path_to_save}\\{filename}_analyzed_{date_time_str}.jpg")
    im.save(f"{image_analyzed_fullpath}")
    is_caries_found = True
    if num_detections == 0:
        is_caries_found = False
    return is_caries_found, image_analyzed_fullpath


if __name__ == "__main__":
    app.run(debug=True)
