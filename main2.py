# importing all the packages needed
import pandas
import torch
import cv2
import os
import json
import streamlit as st
from streamlit_lottie import st_lottie
import numpy as np
import csv
import uuid
print("All packages imported")

def load_animation(filename: str):
    with open(filename,'r') as f:
        json.load(f)

st.title("MarkATT")

main_animation = load_animation(os.path.join("media","main.json"))
st_lottie(main_animation)

grade = st.selectbox(
    "Please choose the grade you want to register for - ",
    ('VI', 'VII', 'VIII', 'IX', 'X', 'XI', 'XII')
)

vi_sections = ['zeus', 'aries']
vii_sections = ['marx', 'seneca', 'plato']
viii_sections = ['Ramanujan', 'Arybhatta']
ix_sections = ['oak', 'pine', 'gulmohar']
x_sections = ['Raman', 'Curie', 'Teressa']

if grade == 'VI':
    sec = st.selectbox('Select Section', vi_sections, key="sec6")
if grade == 'VII':
    sec = st.selectbox('Select Section', vii_sections, key="sec7")
if grade == 'VIII':
    sec = st.selectbox("Select Section", viii_sections, key="sec8")
if grade == 'IX':
    sec = st.selectbox("Select Section", ix_sections, key="sec9")
if grade == 'X':
    sec = st.selectbox("Select Section", x_sections, key="sec10")

paths = {
    "CLASS_FOLDER": f"{grade}_{sec}_folder",
    "CLASS_DATA_PATH": os.path.join(f"{grade}_{sec}_folder", "data"),
    "IMG_SAMPLE_PATH": os.path.join(f"{grade}_{sec}_folder", "data", "images"),
    "LABEL_SAMPLE_PATH": os.path.join(f"{grade}_{sec}_folder", "data", "labels"),
    "CLASS_STATS": os.path.join(f"{grade}_{sec}_folder", "data", "Stats"),
    "CLASS_ATTENDANCE": os.path.join(f"{grade}_{sec}_folder", "attendance_reports")
}

class ClassManager:
    def __init__(self,grade,sec,paths):
        self.grade = grade
        self.sec = sec
        self.paths = paths

    def establish_class_data_paths(self):
        for folders in self.paths.values():
            if not os.path.exists(folders):
                os.mkdir(folders)
                print(f"Folders for {self.grade}-{self.sec} established successfully")
            else:
                print(f"Folders for {self.grade}-{self.sec} are already established")

    def basic_info(self):
        """
        This is the function that will add new students and add their respective info to the data files.
        It will also add the student img samples into the respective paths for further training and model integration.
        """
        form = st.form("Register Student")
        name = form.text_input("Enter Name")
        roll_no = form.number_input("Enter Roll_No")
        email = form.text_input("Enter your email ID")
        submitted = form.form_submit_button("Submit")

        st.title("Take student samples")
        num_samples = 3
        if st.button("Take Student Image Samples"):
            cap = cv2.VideoCapture(0)
            while cap.isOpened():
                for i in range(num_samples):
                    ret,frame = cap.read()
                    cols = st.beta_columns(num_samples)
                    for col in cols:
                        with col:
                            st.write(f"{name}{num_samples}")
                            st.image(frame)

class_page,Register,Attendance,Analytics,Sharing = st.tabs(["Class","Register","Attendance","Analytics","Sharing"])

main_instance = ClassManager(grade,sec,paths)

with class_page:
    st.title(f"Welcome to {grade} {sec}")
    main_instance.establish_class_data_paths()
with Register:
    main_instance.basic_info()

with Attendance:
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # Specify the codec (may vary based on the system)
    out = cv2.VideoWriter("output.mp4", fourcc, 20.0, (640, 480))  # Output video file

    model = torch.hub.load("ultralytics/yolov5",'yolov5s')
    print("Model loaded")
    if st.button("Mark"):
        cap = cv2.VideoCapture(0)

        while True:
            ret,frame = cap.read()
            results = model(frame)
            out.write(np.squeeze(results.render()))

            st.image(np.squeeze(results.render()),channels = 'RGB')

            if cv2.waitKey(0) & 0xFF==ord("q"):
                break
        cap.release()
        out.release()