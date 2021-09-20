import cv2
import pandas as pd 
import glob
import numpy as np
import os.path
##################################
chars = ['A', 'B', 'C', 'D', 'E']
answer_per_question = 5 
questions_per_part = 5
total_parts = 12
ques_each_side = 30
total_question = 60
width = 596
height = 842
#################################
# Import data
img_list = list()
student_data_list = list()
images = [os.path.basename(x) for x in glob.glob("-VNUK-Challenge_2_HAO_UY/DATA/*.png")]

for name in images: 
  img = cv2.imread('-VNUK-Challenge_2_HAO_UY/DATA/'+ name)
  img = cv2.resize(img, (width, height)) 
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
  split = [0,161,241,321,401,478,558]
  if 0 < part < 7:
    img_crop = img[(split[part]):(split[part]+70),150:275]
    return img_crop
  elif 6 < part < 13:
    img_crop = img[(split[part-6]):(split[part-6]+70),350:475] 
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
    image = crop_per_part(answer,x)
    thres = threshold_img(image)
    boxes = split_image (thres)
    pixel_value = check_answer(boxes)
    index = find_index(pixel_value)
    answer_key.append(index)
    x += 1  
  return answer_key[i]

def test_graded(img_index):
  int(img_index)
  graded = []
  x = 1
  while x < 13:
    #Total_part
    answer_index = [] 
    image = crop_per_part(img_list[img_index],x)
    thres = threshold_img(image)
    boxes = split_image (thres)
    pixel_value = check_answer(boxes)
    index = find_index(pixel_value)
    answer_index.append(index)
    answer_key = get_answer(x-1)
    grade = grading(answer_index[0],answer_key)
    graded.append(grade)
    x +=1
  graded = np.concatenate(graded)
  score = score_show(graded)
  return score

def process_chars(index):
  char = []
  for x in range (0,questions_per_part):
    for y in range (0,questions_per_part):
      if index[x] == y:
        index[x] = chars[y]
  return index 

########################################################################
# Ex2: Create CSV file:
df_info = pd.DataFrame(student_data_list, columns=['Student ID','Name','Test Code'])
df_info.to_csv('student_INFO.csv')
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
first_five()
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
all_answer()

######################################################################## 
#Ex5: Generating grading.csv
def correct_questions():
  all_score = list()
  for i in range (0,len(img_list)):     
    all_score.append(test_graded(i))
  return all_score  

def grading_csv():
  correct = correct_questions()
  df_Score = pd.DataFrame({'Score':correct})
  df_ID = pd.DataFrame({'Student ID':df_info['Student ID']})
  frame = [df_ID,df_Score]
  grading = pd.concat(frame,axis=1)
  grading.to_csv('grading.csv')
grading_csv()
########################################################################
#Ex6: Summary which 3 questions are the most 
def all_answer(img_index):
  x = 1
  answer_index = []
  int(img_index)
  while x < 13:
    image = crop_per_part(img_list[img_index],x)
    thres = threshold_img(image)
    boxes = split_image (thres)
    pixel_value = check_answer(boxes)
    index = find_index(pixel_value)
    answer_index.append(index)
    x +=1
  answer_index = np.concatenate(answer_index) 
  return answer_index

def get_answer_all():
  answer = cv2.imread('-VNUK-Challenge_2_HAO_UY/ANSWER/3A.png')
  answer_key = []
  x = 1
  while x < 13:
    image = crop_per_part(answer,x)
    thres = threshold_img(image)
    boxes = split_image (thres)
    pixel_value = check_answer(boxes)
    index = find_index(pixel_value)
    answer_key.append(index)
    x += 1  
  answer_key = np.concatenate(answer_key)
  return answer_key   

answer_key = get_answer_all() 
 
def check_wrong(img_index):
  diff = list()
  for i in range (0,total_question):       
    answer_index = all_answer(img_index)
    if answer_key[i] != answer_index[i]:
      diff.append(i)  
  return diff       

def print_diff():
  total_wrong = list()
  for i in range(0,len(img_list)):
    total_wrong.append(check_wrong(i))
  total_wrong = np.concatenate(total_wrong)
  (counts,unique) = np.unique(total_wrong, return_counts=True)
  diff = np.vstack((counts,unique)).T
  df_diff = pd.DataFrame(diff,columns=['Question','Student Wrong'])
  df_diff = pd.DataFrame.sort_values(df_diff,by='Student Wrong',ascending = False)
  print(df_diff.head(3)) 
print_diff() 
########################################################################
#Ex7: Generating the final result (pass/fail) of the class
def conv_ques():
  conv_correct_questions = [float(x) for x in correct_questions()]
  return conv_correct_questions
def final_result():
  all_result = list()
  correct = conv_ques()
  for i in range (0,len(img_list)):
    if correct[i]>8:
      all_result.append('Pass')
    else:
      all_result.append('Fail')
  return all_result

def result_csv():
  correct = correct_questions()
  result = final_result()
  df_Result = pd.DataFrame(result, columns=['Result'])
  df_Score = pd.DataFrame({'Score':correct})
  frame = [df_info, df_Score, df_Result]
  df_final = pd.concat(frame,axis=1)
  df_final.to_csv('Final Result.csv')
result_csv()
