# Importing all necessary libraries
import cv2
import os
import face_recognition

# Read the video from specified path
#path=r"/home/stegon/Desktop/ExtractImageFromVideoeva-notty-thick-on-thin_720p.mp4"
#path_img=r"C/home/stegon/Desktop/ExtractImageFromVideoEva_Notty.jpg"


fileCfg = open('config.cfg', 'r')

IMMAGINE_PATH=fileCfg.readline().rstrip()
VIDEO_PATH=fileCfg.readline().rstrip()

cam = cv2.VideoCapture(VIDEO_PATH)
img_path=IMMAGINE_PATH


fileCfg.close()

fileCfg = open('config.cfg', 'r')

LISTA_IMMAGINI_PATH=fileCfg.readline().rstrip()
LISTA_VIDEO_PATH=fileCfg.readline().rstrip()
OUTPUT_DIR=fileCfg.readline().rstrip()
LOADER_FRAME_DIR=fileCfg.readline().rstrip()

fileCfg.close()

try:

    # creating a folder named data 
    if not os.path.exists('data'):
        os.makedirs('data')

    # if not created then raise error
except OSError:
    print('Error: Creating directory of data')

# frame 
currentframe = 0
img_file=face_recognition.load_image_file(img_path)
img_encoded=face_recognition.face_encodings(img_file)[0]

while (True):

    # reading from frame 
    ret, frame = cam.read()


    if ret:
        # if video is still left continue creating images 
        name = './data/frame' + str(currentframe) + '.jpg'

        # writing the extracted images
        #coeffDifference=[]
        if  currentframe % 30 ==0:
            print('Creating...' + name)
            try:
                frame_encoded = face_recognition.face_encodings(frame)[0]
                # coeffDifference=face_recognition.compare_faces(frame_encoded, img_encoded);
                exit_attempt = False
                coeffDifference = face_recognition.compare_faces([img_encoded], frame_encoded,0.7)
                if coeffDifference[0]==True:
                    cv2.imwrite(name, frame)

            except:
                print("Nessun match")
                None


        # increasing counter so that it will 
        # show how many frames are created 
        currentframe += 1
    else:
        break

# Release all space and windows once done 
cam.release()
cv2.destroyAllWindows() 