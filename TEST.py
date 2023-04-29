import pickle,time
import cv2
import pyttsx3   #offline lib for tts
import serial  #for serial communication with arduino
import pandas as pd
from datetime import datetime, timedelta


# Read Excel file
df = pd.read_excel('schedule.xlsx')

# Establish connection with arduino
#arduino = serial.Serial(port='COM3', baudrate=9600, timeout=.1)

# Specify the recognizer
face_cascade=cv2.CascadeClassifier('cascades/data/haarcascade_frontalface_alt2.xml')
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('trainner.yml')  # Reading the stored trained model.

# Initiallize speech engine
engine = pyttsx3.init() 

# Loading the label-id relation to get the name of the labels.
with open("label_ids.pickle",'rb') as fr:    
    og_labels=pickle.load(fr)

# Labels are of the form {name:id} we want to invert this form to {id:name}
labels={k:v for v,k in og_labels.items()}

print(labels)

def speak(text):  #fn to convert text to speech
    engine.say(text)
    engine.runAndWait() 

def face_recognition():
    flag_face_recognised = False   #to keep track if the user face is recognized
    flag_face_not_recognised = False

    no_of_adjacent_prediction=0
    no_face_detected=0  #to track the number of times the face is detected
    prev_predicted_name=''   #to keep track of the previously predicted face(w.r.t frame)
    cap=cv2.VideoCapture(0)
    count_frames = total_no_face_detected = 0
    

    while True:
        count_frames+=1
        ret,frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces=face_cascade.detectMultiScale(gray,scaleFactor=1.2,minNeighbors=5)

        for (x,y,w,h) in faces:

            total_no_face_detected+=1
            no_face_detected+=1

            roi_gray=gray[y:y+h,x:x+w]  #roi(region of interest)
            id,confidence=recognizer.predict(roi_gray)

            if(confidence>70):
                font = cv2.FONT_HERSHEY_SIMPLEX
                name=labels[id]
                clr = (0, 255, 0)
                thickness=2
                cv2.putText(frame,name.replace('_',' ').title(),(x,y-5),font,0.8,clr,1,cv2.LINE_AA)
            
                # Check if name matches the name column in the schedule
                if name in df['Name'].tolist():
                    # Get the row where the name matches
                    row = df.loc[df['Name'] == name].iloc[0]
                    # Get the start and end times
                    start_time = row['Start Time']
                    end_time = row['End Time']
                    # Get the current day and time
                    now = datetime.now()
                    current_day = now.strftime('%A')
                    current_time = now.strftime('%H:%M')
                    # Check if the current time and day is within the schedule
                    if start_time <= current_time <= end_time and current_day == row['Day']:
                        # Open the door and welcome the user
                        cv2.putText(frame,f"Welcome {name.replace('_',' ').title()}",(160,460),font,0.8,clr,thickness,cv2.LINE_AA)
                        #arduino.write(b'1')  # Sending signal to arduino to open the door
                        speak(f"Welcome {name.replace('_',' ')}")
                    flag_face_recognised=True
                    flag_face_not_recognised=False
                    prev_predicted_name=name
                    no_of_adjacent_prediction=5

                    # Wait for 5 seconds before resuming recognition
                    time.sleep(5)
                    #arduino.write(b'0')  # Sending signal to arduino to close the door
                else:
                    # Inform the user that they are not allowed to enter
                    cv2.putText(frame,f"You are not allowed to enter at this time",(50,460),font,0.8,(0,0,255),thickness,cv2.LINE_AA)
                    speak(f"I'm sorry {name.replace('_',' ')}, you are not allowed to enter at this time")
                    flag_face_not_recognised=True
                    flag_face_recognised=False
                    no_of_adjacent_prediction=0
                    # Wait for 5 seconds before resuming recognition
                    time.sleep(5)
            else:
                # Inform the user that they are not in the schedule
                cv2.putText(frame,f"You are not in the schedule",(130,460),font,0.8,(0,0,255),thickness,cv2.LINE_AA)
                speak(f"I'm sorry {name.replace('_',' ')}, you are not in the schedule")
                flag_face_not_recognised=True
                flag_face_recognised=False
                no_of_adjacent_prediction=0
                # Wait for 5 seconds before resuming recognition
                time.sleep(5)
                
        else:
            no_of_adjacent_prediction+=1

        if no_of_adjacent_prediction>15:
            no_of_adjacent_prediction=0
        if prev_predicted_name!='':
            cv2.putText(frame,"Identity Not Recognized",(110,460),font,0.8,(0,0,255),thickness,cv2.LINE_AA)
            speak("Identity not recognized, please try again")
            flag_face_not_recognised=True
            flag_face_recognised=False
            prev_predicted_name=''
        # Wait for 5 seconds before resuming recognition
        time.sleep(5)
    
        cv2.imshow('frame',frame)
        if cv2.waitKey(20) & 0xFF == ord('q'):
          break 

cap.release()
cv2.destroyAllWindows()  # release all windows



face_recognition()