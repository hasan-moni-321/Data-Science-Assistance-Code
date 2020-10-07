import os 
import glob 
from tqdm import tqdm 
from PIL import Image, ImageFile 
from joblib import Parallel, delayed 

ImageFile.LOAD_TRUNCATED_IMAGES=True 


def resize_image(image_path, output_folder, resize):
    base_name = os.path.basename(image_path)
    outpath = os.path.join(output_folder, base_name)
    img = Image.open(image_path)
    img = img.resize(
        (resize[1], resize[0]), resample=Image.BILINEAR
    )
    img.save(outpath)



##################################################################
# Resizing train images
##################################################################
imput_folder = ""
output_folder = ""
images = glob.glob(os.path.join(input_folder, "*.jpg"))
Parallel(n_jobs=12)(
    delayed(resize_image)(
        i,
        output_folder,
        (512,512)
    )for i in tqdm(images)
)


####################################################################
# Resizing test images
####################################################################
imput_folder = ""
output_folder = ""
images = glob.glob(os.path.join(input_folder, "*.jpg"))
Parallel(n_jobs=12)(
    delayed(resize_image)(
        i,
        output_folder,
        (512,512)
    )for i in tqdm(images)
)
