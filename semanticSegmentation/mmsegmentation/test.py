from mmseg.apis import MMSegInferencer
# Load models into memory
inferencer = MMSegInferencer(model='deeplabv3plus_r18-d8_4xb2-80k_cityscapes-512x1024')

# images can be a list of image path or np.ndarray
images = ['demo/dog.jpg']

# Save visualized rendering color maps and predicted results
# out_dir is the directory to save the output results, img_out_dir and pred_out_dir are subdirectories of out_dir
# to save visualized rendering color maps and predicted results
inferencer(images, out_dir='outputs', img_out_dir='vis', pred_out_dir='pred')

# this prints out the list of all the model that MMSeg supports
models = MMSegInferencer.list_models('mmseg')
