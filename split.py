import os, random, shutil
import operator
import cv2


def classifyData(fileDir, tarDir):
    label_list = ["angry","disgust","fearful","happy","neutral","sad", "surprise"]
    for dir in os.listdir(fileDir):
        t_dir = os.path.join(fileDir, dir)
        for file in os.listdir(t_dir):
            file_path = os.path.join(t_dir, file)
            label = file.split("-")[1].split("_")[0]
            print(label)
            
            # label_list.append(label)
            for lab in label_list:
                if operator.contains(label,lab):
                    target_path = os.path.join("./original data", tarDir, lab)
                    break
                else:
                    target_path = os.path.join("./original data", tarDir, label)

            # target_path = os.path.join(tarDir, label)
            if not os.path.exists(target_path):
                os.makedirs(target_path)

            target_file = os.path.join(target_path, file)
            shutil.move( file_path, target_file)
    print("classify finish!")
    return

def split(fileDir, tarDir, ratio):
    for dir in os.listdir(fileDir):
        t_dir = os.path.join(fileDir, dir)
        pathDir = os.listdir(t_dir)
        filenumber = len(pathDir)
        picknumber = int(filenumber * ratio)
        sample = random.sample(pathDir, picknumber)
        target_path = os.path.join("./original data",tarDir, dir)
        if not os.path.exists(target_path):
                os.makedirs(target_path)

        for name in sample:
            shutil.move(os.path.join(t_dir, name), target_path)
    os.rename(fileDir,"./original data/train_data")
    print("split finish!")
    return

def resize(src_path, target_path, new_size):
    # Define the desired image size
    # new_size = (256, 256)

    # Loop through the two main folders (train_data and test_data)
    for main_folder in ["train_data", "test_data"]:
        
        # Loop through the subfolders (angry, disgust, fearful, happy, neutral, sad, surprise)
        for emotion_folder in ["angry", "disgust", "fearful", "happy", "neutral", "sad", "surprise"]:
            target_dir = os.path.join( target_path , main_folder, emotion_folder)
            os.makedirs(target_dir )
            # Build the path to the current subfolder
            current_folder = os.path.join(src_path, main_folder, emotion_folder)

            # Loop through the images in the current subfolder
            for filename in os.listdir(current_folder):
                # Read in the current image
                img = cv2.imread(os.path.join(current_folder, filename))

                # Resize the image to the desired size
                resized_img = cv2.resize(img, new_size)

                # Build the path to the resized image
                # resized_filename = os.path.join(current_folder, "resized_" + filename)
                resized_filename = os.path.join(target_dir ,  filename)

                # Save the resized image
                cv2.imwrite(resized_filename, resized_img)
            
    print("resize finish!")

if __name__ == '__main__':
    classifyData("./data/sessions", "./fix_data")
    split("./original data/fix_data", "./test_data", 0.2)
    resize("./original data", "./data", (256, 256))
   
