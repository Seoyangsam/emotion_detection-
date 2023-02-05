import os, random, shutil
import operator

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
                print(lab)
                if operator.contains(label,lab):
                    print("!!")
                    target_path = os.path.join(tarDir, lab)
                    break
                else:
                    target_path = os.path.join(tarDir, label)

            print("00")
            # target_path = os.path.join(tarDir, label)
            if not os.path.exists(target_path):
                os.makedirs(target_path)

            target_file = os.path.join(target_path, file)
            shutil.move( file_path, target_file)
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
    classifyData("./data/sessions", "./fix_data")
    split("./fix_data", "./test_data", 0.2)
   
