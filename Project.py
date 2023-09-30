import cv2
import numpy as np
import face_recognition
import os
from tqdm import tqdm
from datetime import datetime

# импортируем необходимые библиотеки
# далее будут части кода, не влияющие напрямую на алгоритм
# писать комментарии я буду только к основным частям
class bcolors:
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'


path = 'input_images'
path_logs = 'logs'
# эти переменные отвечают за выбор путей ввода и вывода
images = []
classNames = []
mylist = os.listdir(path)
scale_percent = 50
const_scale_time = 2
date = datetime.date(datetime.now())
now = datetime.now().strftime("%H:%M%:%S")
version = "1.2.2"

with open(f"builds.csv", "r") as f:
    build_data = f.readlines()
    for line in build_data:
        entry = line.split(',')
        version_info = entry[0]
        build_info = entry[1]

with open(f"builds.csv", "w") as f:
    if version_info == version:
        f.write(f'{version},{int(build_info) + 1}')
        build_info = int(build_info) + 1
    else:
        f.write(f'{version},1')
        build_info = 1

with open(f"{path_logs}/{date}.csv", "w") as f:
    pass

print(f"Date: {date} Time: {now}")
print(f"Face recognition by Gleb Golubev \nVersion: {version} Build: {build_info}")


# считываем изображения из папки оригиналов
for cls in mylist:
    if cls != '.DS_Store':
        curImg = cv2.imread(f'{path}/{cls}')
        images.append(curImg)
        classNames.append(os.path.splitext(cls)[0])

# переводим изображение в кодировку, переходящую для обработки
def find_encodings(images):
    encode_list = []
    print(f"{bcolors.YELLOW}Compiling images.{bcolors.YELLOW}")
    for img in tqdm(images,colour='green'):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encode_list.append(encode)
    print(f"{bcolors.GREEN}Compiling process finished.{bcolors.GREEN}")
    return encode_list

# отмечаем сходство изображения с камеры и оригинала
def mark_attendance(name):
    global date
    with open(f"{path_logs}/{date}.csv", "r+") as f:
        myDataList = f.readlines()
        nameList = []
        for line in myDataList:
            entry = line.split(',')
            nameList.append(entry[0])
        if name not in nameList:
            now = datetime.now()
            datestring = now.strftime("%H:%M%:%S")
            f.writelines(f'\n{name},{datestring}')


encode_list_known = find_encodings(images)
cap = cv2.VideoCapture(0)


while True:
    success, img = cap.read()
    # считываем изображение с камеры
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    dim = (width, height)
    imgS = cv2.resize(img, dim)
    # масштабируем изображение
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

    # находим лица на фото
    faceCurFrame = face_recognition.face_locations(imgS)
    # кодируем лица
    encodesCurFrame = face_recognition.face_encodings(imgS, faceCurFrame)

    # перебираем все полученные лица
    for encodeFace, faceLoc in zip(encodesCurFrame, faceCurFrame):
        # получаем массив сходств
        matches = face_recognition.compare_faces(encode_list_known, encodeFace)
        # оцениваем, насколько сходство сильное
        faceDist = face_recognition.face_distance(encode_list_known, encodeFace)
        # определяем оригинал, который подходит сходству
        matchInd = np.argmin(faceDist)

        if matches[matchInd]:
            # далее описана обработка изображения с камеры
            name = classNames[matchInd].upper()
            mark_attendance(name)
            y1, x2, y2, x1 = faceLoc
            y1, x2, y2, x1 = y1 * const_scale_time, x2 * const_scale_time, y2 * const_scale_time, x1 * const_scale_time
            # отрисовка прямоугольника вокруг лица и таблички с именем
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.rectangle(img, (x1, y2 - 25), (x2, y2 + 10), (0, 255, 0), cv2.FILLED)
            if x2 - x1 <= 150:
                cv2.putText(img, name, (x1 + 6, y2 + 4), cv2.FONT_ITALIC, 0.5, (0, 0, 0), 2)
            elif x2 - x1 <= 200:
                cv2.putText(img, name, (x1 + 6, y2 + 4), cv2.FONT_ITALIC, 0.6, (0, 0, 0), 2)
            elif x2 - x1 <= 250:
                cv2.putText(img, name, (x1 + 6, y2 + 4), cv2.FONT_ITALIC, 0.8, (0, 0, 0), 2)
            else:
                cv2.putText(img, name, (x1 + 6, y2 + 4), cv2.FONT_ITALIC, 0.95, (0, 0, 0), 2)
        else:
            # то же самое, только если сходство не найдено
            name = "Unknown"
            y1, x2, y2, x1 = faceLoc
            y1, x2, y2, x1 = y1 * const_scale_time, x2 * const_scale_time, y2 * const_scale_time, x1 * const_scale_time
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.rectangle(img, (x1, y2 - 25), (x2, y2 + 10), (0, 0, 255), cv2.FILLED)
            if x2 - x1 <= 150:
                cv2.putText(img, name, (x1 + 6, y2 + 4), cv2.FONT_ITALIC, 0.5, (0, 0, 0), 2)
            elif x2 - x1 <= 200:
                cv2.putText(img, name, (x1 + 6, y2 + 4), cv2.FONT_ITALIC, 0.6, (0, 0, 0), 2)
            elif x2 - x1 <= 250:
                cv2.putText(img, name, (x1 + 6, y2 + 4), cv2.FONT_ITALIC, 0.8, (0, 0, 0), 2)
            else:
                cv2.putText(img, name, (x1 + 6, y2 + 4), cv2.FONT_ITALIC, 0.95, (0, 0, 0), 2)

    cv2.imshow('Webcam', img)
    quit_key = cv2.waitKey(1)
    # обработка выхода из программы
    if quit_key == ord('x') or quit_key == ord('X'):
        cv2.destroyAllWindows()
        break
