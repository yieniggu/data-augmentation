# Script que realiza distintas operaciones en imagenes para aumentar la data disponible

# Import libraries
import imageio
import argparse
import os
import sys
import imgaug as ia
import imgaug.augmenters as iaa
from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage
import glob
import xml.etree.ElementTree as ET
import numpy as np
import pandas as pd

def gaussian_blur(images_path, annotations_path, imgs_output_path, xmls_output_path):
    # Init augmenter
    gaussian_blur = iaa.GaussianBlur(sigma=(0.5, 3.0))

    for image_path in images_path:

        # Abrimos la imagen
        image = imageio.imread(image_path)
    
        # Abrimos el xml
        xml_tree = read_xml(annotations_path, image_path)

        # Agregamos el ruido
        image_augmented = gaussian_blur.augment_image(image)

        # Obtain image filename
        filename = image_path.split('/')[-1].split('.')[0]
        #print("filename: ", filename)

        # Save augmented image and annotations
        image_augmented_path = imgs_output_path + filename + "_blured.jpg"
        xml_augmented_path = xmls_output_path + filename + "_blured.xml"
        
        print("\n[INFO] Writing augmented image: {} to {}".format(filename + ".jpg", image_augmented_path))
        print("[INFO] Writing augmented xml: {} to {}".format(filename + ".xml", xml_augmented_path))

        imageio.imwrite(image_augmented_path, image_augmented)
        xml_tree.write(xml_augmented_path)

# Añade ruido gaussiano a la imagen
def add_noise(images_path, annotations_path, imgs_output_path, xmls_output_path):
    # Inicializamos el augmenter
    gaussian_noise=iaa.AdditiveGaussianNoise(10, 20)

    for image_path in images_path:

        # Abrimos la imagen
        image = imageio.imread(image_path)
    
        # Abrimos el xml
        xml_tree = read_xml(annotations_path, image_path)

        # Agregamos el ruido
        image_augmented = gaussian_noise.augment_image(image)

        # Obtain image filename
        filename = image_path.split('/')[-1].split('.')[0]
        #print("filename: ", filename)

        # Save augmented image and annotations
        image_augmented_path = imgs_output_path + filename + "_noised.jpg"
        xml_augmented_path = xmls_output_path + filename + "_noised.xml"
        
        print("\n[INFO] Writing augmented image: {} to {}".format(filename + ".jpg", image_augmented_path))
        print("[INFO] Writing augmented xml: {} to {}".format(filename + ".xml", xml_augmented_path))

        imageio.imwrite(image_augmented_path, image_augmented)
        xml_tree.write(xml_augmented_path)

        

# Rota la imagen en todos los angulos disponibles y las guarda
def rotate(images_path, annotations_path, imgs_output_path, xmls_output_path):
    angles = np.array([90, 180, 270], dtype=int)

    for image_path in images_path:
        # Abrimos la imagen
        image = imageio.imread(image_path)
    
        # Obtenemos las bb de la imagen
        bbs_df = get_bbs(annotations_path, image_path)

        # Obtenemos los valores de clases y bboxes
        classes = bbs_df.iloc[:, 0:1].values
        bbs_values = bbs_df.iloc[:, 1:].values

        # Init bbs on imgaug format
        bbs = init_bbs(bbs_values.astype(int), classes)

        # Rotamos la imagen en todos los angulos
        for angle in angles:
            # Init augmenter
            rotate=iaa.Affine(rotate=(angle))
            image_augmented, bbs_augmented =rotate (image=image, bounding_boxes=bbs)

            # Create the new xml
            #print("img_augmented_shape: ", image_augmented.shape)
            augmented_xml = augment_xml(annotations_path, image_path, bbs_augmented, image_augmented.shape)

            # Obtain image filename
            filename = image_path.split('/')[-1].split('.')[0]
            #print("filename: ", filename)

            # Save augmented image and annotations
            image_augmented_path = "{}{}_rotated{}.jpg".format(imgs_output_path, filename, angle)
            xml_augmented_path = "{}{}_rotated{}.xml".format(xmls_output_path, filename, angle)
            
            print("\n[INFO] Writing augmented image: {} to {}".format(filename + ".jpg", image_augmented_path))
            print("[INFO] Writing augmented xml: {} to {}".format(filename + ".xml", xml_augmented_path))

            imageio.imwrite(image_augmented_path, image_augmented)
            augmented_xml.write(xml_augmented_path)


# Flip horizontal a cada imagen
def flip_horizontal(images_path, annotations_path, imgs_output_path, xmls_output_path):
    # Inicializamos el augmenter
    flip_hr = iaa.Fliplr(p=1.0)
    for image_path in images_path:
        
        # Abrimos la imagen
        image = imageio.imread(image_path)

        # Obtenemos las bb de la imagen
        bbs_df = get_bbs(annotations_path, image_path)
        #print(bbs_df.head())

        # Obtenemos los valores de clases y bboxes
        classes = bbs_df.iloc[:, 0:1].values

        bbs_values = bbs_df.iloc[:, 1:].values

        #print("Classes: \n{} - \nType: {}".format(classes, type(classes)))
        #print("bbs_values: \n{} \nType: {}".format(bbs_values, type(bbs_values)))

        # Inicializamos las bbs
        bbs = init_bbs(bbs_values.astype(int), classes)
        #print("bbs: \n{} \nType: {}\n".format(bbs, type(bbs)))

        # Apply augmentation
        image_augmented, bbs_augmented = flip_hr(image= image, bounding_boxes=bbs)
        #print("bbs_augmented: ", bbs_augmented)

        # Create the new xml
        #print("img_augmented_shape: ", image_augmented.shape)
        augmented_xml = augment_xml(annotations_path, image_path, bbs_augmented, image_augmented.shape)

        # Obtain image filename
        filename = image_path.split('/')[-1].split('.')[0]
        #print("filename: ", filename)

        # Save augmented image and annotations
        image_augmented_path = imgs_output_path + filename + "_flipped.jpg"
        xml_augmented_path = xmls_output_path + filename + "_flipped.xml"
        
        print("\n[INFO] Writing augmented image: {} to {}".format(filename + ".jpg", image_augmented_path))
        print("[INFO] Writing augmented xml: {} to {}".format(filename + ".xml", xml_augmented_path))

        imageio.imwrite(image_augmented_path, image_augmented)
        augmented_xml.write(xml_augmented_path)

def init_bbs(bbs_values, classes):
    bbs_list = []

    for item in range(classes.shape[0]):
        new_bb = BoundingBox(x1=bbs_values[item][0], y1=bbs_values[item][2], x2=bbs_values[item][1], y2=bbs_values[item][3], label=classes[item][0])
        bbs_list.append(new_bb)

    return bbs_list

def get_bbs(annotations_path, img_path):
    # Cargamos los tags de objetos en cada imagen
    objects = read_xml_objects(annotations_path, img_path)

    # Definimos las columnas a utilizar por el dataframe
    columns = [["name", "xmin", "xmax", "ymin", "ymax"]]
    bbs_df = pd.DataFrame(columns=columns)

    for obj in objects:
        bb = obj.find('bndbox')
        data = np.array([obj.find('name').text, int(bb.find('xmin').text), int(bb.find('xmax').text), int(bb.find('ymin').text), int(bb.find('ymax').text)]).reshape(1,-1)
        temp_df = pd.DataFrame(data, columns=columns)
        bbs_df = bbs_df.append(temp_df)
    
    return bbs_df

def read_xml_objects(annotations_path, img_path):
    # Separamos los string por / y sacamos el ultimo
    img_name = img_path.split('/')[-1]
    
    # Obtenemos el nombre del archivo sin extension
    filename, _ = os.path.splitext(img_name)
    # Definimos la ruta del archivo xml
    xml_name = annotations_path + filename + ".xml"
    
    # Retornamos todos tags con objects
    return ET.parse(xml_name).getroot().findall('object')
    

def read_images(img_path):
    # Obtenemos las rutas de los archivos
    images_path = [f for f in glob.glob(img_path + "*.jpg", recursive=True)]

    return images_path

def augment_xml(annotations_path, image_path, bbs_augmented, img_shape):
    # Read the xml
    xml_tree = read_xml(annotations_path, image_path)

    # Get the root
    xml_root = xml_tree.getroot()

    # Lets remove size element and update it
    xml_root.remove(xml_root.find("size"))

    # Creates size subelement
    size_subelement = ET.SubElement(xml_root, "size")
    width_subelement = ET.SubElement(size_subelement, "width")
    width_subelement.text = str(img_shape[1])

    height_subelement = ET.SubElement(size_subelement, "height")
    height_subelement.text = str(img_shape[0])

    depth_subelement = ET.SubElement(size_subelement, "depth")
    depth_subelement.text = str(img_shape[2])

    # Lets also remove object elements
    for objectt in xml_root.findall("object"):
        xml_root.remove(objectt)

    # Creates the new object annotations
    for bb_augmented in bbs_augmented:
        bb_data = vars(bb_augmented)
        
        #print("bb_augmented: ", bb_data)
        object_subelement = ET.SubElement(xml_root, "object")
        
        # Define name tag & value
        name_subelement = ET.SubElement(object_subelement, "name")
        name_subelement.text = bb_data["label"]

        # Define pose tag & value
        pose_subelement = ET.SubElement(object_subelement, "pose")
        pose_subelement.text = "Unspecified"

        # Define truncated tag & value
        truncated_subelement = ET.SubElement(object_subelement, "truncated")
        truncated_subelement.text = "0"

        # Define difficult tag & value
        difficult_subelement = ET.SubElement(object_subelement, "difficult")
        difficult_subelement.text = "0"

        # Define new bb
        bb_subelement = ET.SubElement(object_subelement, "bndbox")
        
        xmin_subelement = ET.SubElement(bb_subelement, "xmin")
        xmin_subelement.text = str(bb_data["x1"])

        ymin_subelement = ET.SubElement(bb_subelement, "ymin")
        ymin_subelement.text = str(bb_data["y1"])

        xmax_subelement = ET.SubElement(bb_subelement, "xmax")
        xmax_subelement.text = str(bb_data["x2"])

        ymax_subelement = ET.SubElement(bb_subelement, "ymax")
        ymax_subelement.text = str(bb_data["y2"])


    return xml_tree


def read_xml(annotations_path, img_path):
    # Separamos los string por / y sacamos el ultimo
    img_name = img_path.split('/')[-1]
    
    # Obtenemos el nombre del archivo sin extension
    filename, _ = os.path.splitext(img_name)
    # Definimos la ruta del archivo xml
    xml_name = annotations_path + filename + ".xml"

    # Retornamos todos tags con objects
    return ET.parse(xml_name)

#def save(output_path):


def main():
    # Initiate argument parser
    parser = argparse.ArgumentParser(
        description="Image Augmentation operations")
    parser.add_argument("-i",
                        "--imagesFolder",
                        help="Path to the folder where the input images files are stored",
                        type=str)
    parser.add_argument("-a",
                        "--annotationsFolder",
                        help="Path to the folder where the input xmls files are stored", type=str)
    parser.add_argument("-io",
                        "--imagesOutputFolder",
                        help="Name of output to save the images", type=str)
    parser.add_argument("-lo",
                        "--labelsOutputFolder",
                        help="Name of output to save the labels", type=str)
    parser.add_argument("-aug",
                        "--augmentation",
                        help="Augmentation to perform")
    args = parser.parse_args()

    # creating a folder named data 
    print ('[START] Creating directory of data') 
    if not os.path.exists(args.imagesOutputFolder): 
        os.makedirs(args.imagesOutputFolder) 
    else:
        print ('[INFO] The specified img folder already exists') 
 
    if not os.path.exists(args.labelsOutputFolder): 
        os.makedirs(args.labelsOutputFolder) 
    else:
        print ('[INFO] The specified XMLS folder already exists')

    print("[INFO] Image Directory: ", args.imagesFolder)
    print("[INFO] Images Output folder: ", args.imagesOutputFolder)
    print("[INFO] Labels Output folder: ", args.labelsOutputFolder)

    # Leemos las imagenes
    images = read_images(args.imagesFolder)

    if args.augmentation == 'flip':
        flip_horizontal(images, args.annotationsFolder, args.imagesOutputFolder, args.labelsOutputFolder)
    elif args.augmentation == 'noise':
        add_noise(images, args.annotationsFolder, args.imagesOutputFolder, args.labelsOutputFolder)
    elif args.augmentation == 'rotate':
        rotate(images, args.annotationsFolder, args.imagesOutputFolder, args.labelsOutputFolder)
    elif args.augmentation == 'blur':
        gaussian_blur(images, args.annotationsFolder, args.imagesOutputFolder, args.labelsOutputFolder)
    else:
        print("[ERROR] Invalid augmentation type")
        sys.exit()

    print("\n[FINISH] All images processed succesfully")

if __name__ == '__main__':
    main()

