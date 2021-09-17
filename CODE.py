import cv2
import pandas as pd 
import matplotlib.pyplot as plt

height = 600 
width = 463

character_table = ['A', 'B', 'C', 'D', 'E']
answer_per_question = 5 
questions_per_part = 5
total_parts = 6
ques_each_side = 30
total_question = 60

statistical_table = [0] * total_question

green = (0, 255, 0) # green color
red = (0, 0, 255) # red color
white = (255, 255, 255) # white color