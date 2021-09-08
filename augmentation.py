import numpy as np
import albumentations as A

def transform(image):
    
    if image.dtype != "uint8":
        image = (255*image).astype("uint8")
    
    aug = A.Compose([
        A.FancyPCA (alpha=0.08, always_apply=False, p=1),
    ])
    
    image = aug(image=image)['image'].astype("float32")

    return image