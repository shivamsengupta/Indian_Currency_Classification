from ultralytics import YOLO
import cv2

def main():

    model=YOLO("best.pt")
    camera = cv2.VideoCapture(0)
    while True: 
        
        ret, frame = camera.read()
        if not ret:
            break
        results = model(frame)
        best_class_key=results[0].probs.top1
        names_dict=results[0].names
        class_index = names_dict[best_class_key]
        p_val=round(results[0].probs.top1conf.item(),2)

        cv2.putText(frame,f"Class: {class_index}, Accuracy: {p_val*100}%" , (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        cv2.imshow("Camera", frame)

        if cv2.waitKey(1) & 0xFF == 27:  # Press 'Esc' key to exit
            break
    camera.release()
    cv2.destroyAllWindows()
       

if __name__ == "__main__":
    main()
