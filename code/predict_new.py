from ultralytics import YOLO
import cv2
from ultralytics.utils.plotting import Annotator
import matplotlib.pyplot as plt
import math

def calculate_distance(x1, y1, x2, y2):
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

# Initialize the YOLO model
model = YOLO('best.pt')

# cap = cv2.VideoCapture(r"shuttle_19.jpg")
# MVI_9716.MP4_Rendered_001

def predict_height(file_path):
    cap = cv2.VideoCapture(file_path)
    cap.set(3, 640)
    cap.set(4, 480)

    while True:
        _, img = cap.read()
        c1x1, c1y1, c1x2, c1y2, c1mx, c1my = 0, 0, 0, 0, 0, 0
        c2x1, c2y1, c2x2, c2y2, c2mx, c2my = 0, 0, 0, 0, 0, 0
        nx1, ny1, nx2, ny2, nmx, nmy = 0, 0, 0, 0, 0, 0
        sx1, sy1, sx2, sy2, smx, smy = 0, 0, 0, 0, 0, 0
        
        results = model.predict(img)

        for r in results:
            
            annotator = Annotator(img)
            
            boxes = r.boxes
            for box in boxes:
                
                b = box.xyxy[0]  # get box coordinates in (left, top, right, bottom) format
                c = box.cls
                # print(c)
                annotator.box_label(b, model.names[int(c)])

                if(model.names[int(c)] == "c1"):
                    c1x1,c1y1,c1x2,c1y2 = b
                    c1mx,c1my=((int(c1x1) + int(c1x2)) // 2, (int(c1y1) + int(c1y2)) // 2)
                    # cv2.circle(img, (int(c1mx), int(c1my)), 5, (255, 0, 0), -1)  # Blue point at the middle
                
                elif(model.names[int(c)] == "c2"):
                    c2x1,c2y1,c2x2,c2y2 = b
                    c2mx,c2my=((int(c2x1) + int(c2x2)) // 2, (int(c2y1) + int(c2y2)) // 2)
                    # cv2.circle(img, (int(c2mx), int(c2my)), 5, (255, 0, 0), -1)  # Blue point at the middle

                elif(model.names[int(c)] == "net_level"):
                    nx1,ny1,nx2,ny2 = b
                    nmx,nmy=((int(nx1) + int(nx2)) // 2, int(ny1))
                    # cv2.circle(img, (int(nmx), int(nmy)), 5, (255, 0, 0), -1)  # Blue point at the middle

                elif(model.names[int(c)] == "shuttle"):
                    sx1,sy1,sx2,sy2 = b
                    smx,smy=((int(sx1) + int(sx2)) // 2, (int(sy1) + int(sy2)) // 2)
                    # cv2.circle(img, (int(smx), int(smy)), 5, (255, 0, 0), -1)  # Blue point at the middle
            
            distance_between_c1_c2=calculate_distance(c1mx, c1my, c2mx, c2my)
            by_one_pixel = 396/distance_between_c1_c2

            distance_between_net_shuttle=calculate_distance(nmx, nmy, smx, smy)
            actual_distance_between_net_shuttle = distance_between_net_shuttle * by_one_pixel
            
        limit = 50
        if nmx - limit < smx and smx < nmx + limit:      
            img = annotator.result() 
            shot_result = "" 

            if smx>0 and smy>0:
                cv2.line(img, (nmx, nmy), (smx, smy), (0, 0, 255), 2)
                text = f"Distance: {actual_distance_between_net_shuttle:.2f} cm"
                cv2.putText(img, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

                height_limit = 120

                if actual_distance_between_net_shuttle <= height_limit:
                    shot_result ="Not attackable"
                    # cv2.putText(img, "Not attackable", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 255, 0), 2, cv2.LINE_AA)
                else:
                    shot_result ="Attackable"
                    # cv2.putText(img, "Attackable", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 255), 2, cv2.LINE_AA)

            cv2.putText(img, shot_result, (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 255), 2, cv2.LINE_AA)

            # Convert BGR to RGB for Matplotlib display
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Display using Matplotlib
            plt.imshow(img_rgb)
            plt.axis('off') 
            plt.show()


    cap.release()
    return shot_result, img_rgb, actual_distance_between_net_shuttle

predict_height(r"MVI_9716.MP4_Rendered_002.mp4")
