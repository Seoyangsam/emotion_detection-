import os, random, shutil

def ClassifyData(fileDir, tarDir):
    for dir in os.listdir(fileDir):
        t_dir = os.path.join(fileDir, dir)
        for file in os.listdir(t_dir):
            file_path = os.path.join(t_dir, file)
            label = file.split("-")[1].split("_")[0]
            target_path = os.path.join(tarDir, label)

            if not os.path.exists(target_path):
                os.makedirs(target_path)

            shutil.move( file_path, target_path)
    print("finish!")
    return

def split(fileDir, tarDir, ratio):
    for dir in os.listdir(fileDir):
        t_dir = os.path.join(fileDir, dir)
        pathDir = os.listdir(t_dir)
        filenumber = len(pathDir)
        picknumber = int(filenumber * ratio)
        sample = random.sample(pathDir, picknumber)
        target_path = os.path.join(tarDir, dir)
        if not os.path.exists(target_path):
                os.makedirs(target_path)

        for name in sample:
            shutil.move(os.path.join(t_dir, name), target_path)
    os.rename(fileDir,"train_data")
    return



if __name__ == '__main__':
    ClassifyData("./data/sessions", "./fix_data")
    split("./fix_data", "./test_data", 0.2)
   
