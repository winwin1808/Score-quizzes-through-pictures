from os import sep
import cv2
import pandas as pd 
import matplotlib.pyplot as plt
import glob
import imutils
import numpy as np
import os.path
import csv

chars = ['A', 'B', 'C', 'D', 'E']
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
# os.chdir("D:\LEARN\Python\Problemset_2\-VNUK-Challenge_2_HAO_UY\DATA")
# path = glob.glob("DATA/*.png")

# images = glob.glob("DATA/*.png")
images = [os.path.basename(x) for x in glob.glob("-VNUK-Challenge_2_HAO_UY/DATA/*.png")]
for name in images: 
  img = cv2.imread('-VNUK-Challenge_2_HAO_UY/DATA/'+ name)
  img = cv2.resize(img, (width, height)) 
  # img = img[70:,:]
  img_list.append(img)
  
  name = name.replace('.png', '').split('_')
  student = [name[0], name[1], name[2]]
  student_data_list.append(name)

################################

def show_images(titles, images, wait=True):
    for (title, image) in zip(titles, images):
        cv2.imshow(title, image)
    if wait:
        cv2.waitKey(0)
        cv2.destroyAllWindows()


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
  score = (sum(grading)/total_question)*10
  score = "%.2f" % score
  return score

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
    x += 1  
  return answer_key[i]

def process_chars(index):
  char = []
  for x in range (0,questions_per_part):
    for y in range (0,questions_per_part):
      if index[x] == y:
        index[x] = chars[y]
  return index 


def test_graded(img_index):
  graded = []
  char =[]
  x = 1
  while x < 13:
    answer_index = [] 
    thres = threshold_img(crop_per_part(img_list[img_index],x))
    boxes = split_image (thres)
    pixel_value = check_answer(boxes)
    index = find_index(pixel_value)
    answer_index.append(index)
    # char.append(process_chars(answer_index[0]))
    grade = grading(answer_index[0],get_answer(x-1))
    graded.append(grade)
    x +=1
  # process_chars(answer_index)
  graded = np.concatenate(graded)
  # print(graded)
  score = score_show(graded)
  return score


def correct_questions():
  all_score = list()
  for i in range (0,len(img_list)):     
    all_score.append(test_graded(i))
  return all_score  
########################################################################
# add_dataframe()
########################################################################
# Ex2: Create CSV file:
df_info = pd.DataFrame(student_data_list, columns=['Student ID','Name','Test Code'])
df_info.to_csv('student_INFO.csv', index=False, sep=';')
########################################################################
#Ex3: Generating the first 5 answers of one student:
def first_five():
  char = []
  answer_index = [] 
  thres = threshold_img(crop_per_part(img_list[0],1))
  boxes = split_image (thres)
  pixel_value = check_answer(boxes)
  index = find_index(pixel_value)
  answer_index.append(index)
  char.append(process_chars(answer_index[0]))
  # grade = grading(answer_index[0],get_answer(1))
  # grade_.append(grade)
  # answer_index = np.concatenate(answer_index)
  # grade = grading(answer_index,get_answer())
  # score = score_show(grade)
  print('The first 5 answers of the first student:')
  print(str(char))
# first_five()
######################################################################## 
# Ex4: Generating all answers of one student:
def all_answer():
  char =[]
  x = 1
  while x < 13:
    answer_index = []
    thres = threshold_img(crop_per_part(img_list[0],x))
    boxes = split_image (thres)
    pixel_value = check_answer(boxes)
    index = find_index(pixel_value)
    answer_index.append(index)
    char.append(process_chars(answer_index[0]))
    x +=1

  print('All answers of the first student:')
  print(str(char))
# all_answer()

######################################################################## 
#Ex5: Generating grading.csv
correct_questions = correct_questions()
def grading_csv():
  df_Score = pd.DataFrame({'Score':correct_questions})
  df_ID = pd.DataFrame({'Student ID':df_info['Student ID']})
  frame = [df_ID,df_Score]
  grading = pd.concat(frame,axis=1)
  grading.to_csv('grading.csv')
grading_csv()
########################################################################
#Ex6: Summary which 3 questions are the most 

########################################################################
#Ex7: Generating the final result (pass/fail) of the class
conv_correct_questions = [float(x) for x in correct_questions]
def final_result():
  all_result = list()
  for i in range (0,len(img_list)):
    if conv_correct_questions[i]>8:
      all_result.append(print('Pass'))
    else:
      all_result.append(print('Fail'))
  return all_result

def result_csv():
  result = final_result()
  df_Result = pd.DataFrame(result, columns=['Result'])
  df_Score = pd.DataFrame({'Score':correct_questions})
  frame = [df_info, df_Score, df_Result]
  df_final = pd.concat(frame,axis=1)
  df_final.to_csv('Final Result.csv')
result_csv()
