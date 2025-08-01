import pandas as pd
import os
from PIL import Image
from tqdm import tqdm

data_dir=os.path.join("data","New Plant Diseases Dataset(Augmented)","New Plant Diseases Dataset(Augmented)","train")

def load_dataset_info(data_dir):
    data=[]

    for class_folder in tqdm(os.listdir(data_dir)):
        class_path=os.path.join(data_dir,class_folder)

        if not os.path.isdir(class_path):
            continue

        for img_file in os.listdir(class_path):
            img_path=os.path.join(class_path,img_file)

            try:
                img=Image.open(img_path)
                img.verify()

            except:
                continue

            data.append({
                "image_path": img_path,
                "label":class_folder
        })
    return pd.DataFrame(data)

if __name__=="__main__":
    df=load_dataset_info(data_dir)
    print("Total images: ",len(df))
    print("Sample Rows")
    print(df.sample(5))