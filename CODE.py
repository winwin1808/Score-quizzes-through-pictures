import cv2
import pandas as pd 
import matplotlib.pyplot as plt
import glob
import imutils
import numpy as np

character_table = ['A', 'B', 'C', 'D', 'E']
answer_per_question = 5 
questions_per_part = 5
total_parts = 12
ques_each_side = 30
total_question = 60


green = (0, 255, 0) # green color
red = (0, 0, 255) # red color
white = (255, 255, 255) # white color

width = 596
height = 842

#################################
# Import data

# data = ['DATA/'+ 
# '20012311_TranDuc_3A.png',
# '20012312_QuyLan_3A.png',
# '20012313_HoangGiaUy_3A.png',
# '20012317_NguyenQuangHao_3A.png',
# '20012318_NguyenVanC_3A.png',
# '20012319_NguyenThiNo_3A.png']

img_list = list()
student_data_list = list()

path = glob.glob("DATA/*.png")

for name in path: 
  img = cv2.imread(name)
  img = cv2.resize(img, (width, height))
  # img = img[70:,:]
  img_list.append(img)
  
  name = name.replace('.png', '').split('_')
  student = [name[0], name[1], name[2]]  
  student_data_list.append(name)

################################

ANSWER_KEY = {0: 2, 1: 1, 2: 2, 3: 3, 4: 1}

def show_images(titles, images, wait=True):
    """Display multiple images with one line of code"""

    for (title, image) in zip(titles, images):
        cv2.imshow(title, image)

    if wait:
        cv2.waitKey(0)
        cv2.destroyAllWindows()

def generateColumns(): 
  columns = ['Student ID','Name','Test Code']
  for i in range(total_question): 
    columns.append('Question: '+str((i+1)))
  columns.append('Score')
  return columns

def grid(): 
  grid = [] 
  for i in range(total_question): 
    grid.append([])
    for j in range(answer_per_question): 
      grid[i].append(0)
  return grid

def write_csv_file(): 
  if len(student_data_list) == 0: 
    return False
  students = pd.DataFrame(student_data_list, columns=['Student ID','Name','Test Code'])
  students.to_csv('student.csv', index=False, sep=';')

def threshold_img(img):
  gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  _, thresh = cv2.threshold(gray, 250, 255, cv2.THRESH_BINARY_INV)
  return thresh

def crop_per_part(img,part):

  part = int(part)
  dis = 70
  xa= 10
  h = 237
  
  if part == 1:
    img_crop = img[(h+int(dis*(part-1))):(h+int(dis*part)),150:275]
    return img_crop
  elif 1 < part < 7:
    img_crop = img[(h+int(dis*(part-1))+xa):(h+int(dis*part)+xa),150:275]  
    return img_crop
  elif part == 7:
    img_crop = img[(h+int(dis*(part-7))):(h+int(dis*(part-6))),350:475]
    return img_crop
  elif 6 < part < 14:
    img_crop = img[(h+int(dis*(part-7))+xa):(h+int(dis*(part-6))+xa),350:475]
    return img_crop
  else:
    return img

def split_image(image):
    rows = np.vsplit(image,questions_per_part)
    boxes = []
    for row in rows:
        # split each row vertically (column-wise)
        cols = np.hsplit(row, answer_per_question)  
        for box in cols:  
            boxes.append(box)
    return boxes

def check_answer(boxes):
  myPixelVal = np.zeros((questions_per_part, answer_per_question))
  countC = 0
  countR = 0
  for image in boxes:
    totalPixels = cv2.countNonZero(image)
    myPixelVal[countR][countC] = totalPixels
    countC += 1
    if (countC == answer_per_question):
      countR +=1 
      countC = 0
  return myPixelVal    

def find_index(myPixelVal):
  index = list()

  for x in range (0,questions_per_part):
    arr = myPixelVal[x]
    index_value = np.where(arr == np.amax(arr))
    index.append(index_value[0][0])
    str(index)
  return index

def grading(index,answer):
  grading = []
  
  for x in range (0,questions_per_part):
    if answer[x] == index[x]:
      grading.append(1)
    else:
      grading.append(0)
  return grading

def score_show(grading):
  score = (sum(grading)/questions_per_part)*100
  return score
# # boxes = img_list[0].copy()
# boxes = split_image(threshold_img(crop_per_part(img_list[0],1)))
# # cv2.imshow('Test',boxes[20])
# Pixel_value = check_answer(boxes)
# index = find_index(Pixel_value)
# grading(index,ANSWER_KEY)
# print(score_show(grading))

def get_answer(i):
  answer = cv2.imread('-VNUK-Challenge_2_HAO_UY/ANSWER/3A.png')
  answer_key = []
  x = 1
  while x < 13:
    thres = threshold_img(crop_per_part(answer,x))
    boxes = split_image (thres)
    pixel_value = check_answer(boxes)
    index = find_index(pixel_value)
    answer_key.append(index)
    str(answer_key)
    x += 1  
  # answer_key = np.hstack(answer_key)
  # answer_key = np.hstack(answer_key)
  return answer_key[i]


# print(get_answer(1))
# img = cv2.imread('-VNUK-Challenge_2_HAO_UY/ANSWER/3A.png') 
# cv2.imshow('answer',img)
# cv2.waitKey(0)
# boxes = split_image(threshold_img(crop_per_part(img_list[0]),1))
# answer_check(crop_per_part(img_list[0],21))
# show_images(['Part_1'], [threshold_img(crop_per_part(img_list[0],1))])

# write_csv_file()
########################################################################
# Ex2: Create CSV file:
# write_csv_file()
########################################################################
#Ex3: Generating the first 5 answers of one student:
def test_graded():
  grade_ = []
  x = 1
  while x < 13:
    answer_index = []
    thres = threshold_img(crop_per_part(img_list[4],x))
    boxes = split_image (thres)
    pixel_value = check_answer(boxes)
    index = find_index(pixel_value)
    answer_index.append(index)
    grade = grading(answer_index[0],get_answer(x-1))
    grade_.append(grade)
    x +=1
  # answer_index = np.concatenate(answer_index)
  # grade = grading(answer_index,get_answer())
  score = score_show(grade)
  return score
#################################
# answer_index = []
# thres = threshold_img(crop_per_part(img,1))
# boxes = split_image (thres)
# pixel_value = check_answer(boxes)
# index = find_index(pixel_value)
# answer_index.append(index)
# grade = grading(answer_index[0],get_answer(0))
# # grade = grading(answer_index,get_answer(0))
# print(answer_index[0])
# print(get_answer(0))
# print(grade)
# print(score_show(grade))
print(test_graded())

  
  