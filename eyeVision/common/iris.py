import cv2 as cv
import numpy as np
import mediapipe as mp

class Iris:
  
  def __init__(self) -> None:
     self.left_eye = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385,384, 398]
     self.left_iris = [474, 475, 476, 477]
     self.right_eye = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161 , 246]  
     self.right_iris = [469, 470, 471, 472] 
     self.mp_face_mesh = mp.solutions.face_mesh
     self.cap = cv.VideoCapture(1)

  def Detection(self) -> np.array:
    with self.mp_face_mesh.FaceMesh(max_num_faces = 1, refine_landmarks = True, 
        min_detection_confidence = 0.5, min_tracking_confidence = 0.5) as face_mesh:
      
      while True:
          ret, frame = self.cap.read()
          
          key = cv.waitKey(1)
          
          if key == ord('q') or not ret:
              break
          
          frame = cv.flip(frame, 1) 
          rgb_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
          img_h, img_w = frame.shape[:2]
          results = face_mesh.process(rgb_frame)
          
          if results.multi_face_landmarks:
            mesh_points = np.array([np.multiply([p.x, p.y], [img_w, img_h]).astype(int) for p in results.multi_face_landmarks[0].landmark])
            
            cv.polylines(frame, [mesh_points[self.left_eye]], True, (0, 255, 0), 1, cv.LINE_AA)
            cv.polylines(frame, [mesh_points[self.right_eye]], True, (0, 255, 0), 1, cv.LINE_AA)
            
            (l_cx, l_cy), l_radius = cv.minEnclosingCircle(mesh_points[self.left_iris])
            (r_cx, r_cy), r_radius = cv.minEnclosingCircle(mesh_points[self.right_iris])
            center_left = np.array([l_cx, l_cy], dtype = np.int32)
            center_right = np.array([r_cx, r_cy], dtype = np.int32)
            
            cv.circle(frame, center_left, int(l_radius), (255, 0, 255), 1, cv.LINE_AA)
            cv.circle(frame, center_right, int(r_radius), (255, 0, 255), 1, cv.LINE_AA)

            print(center_left, l_radius)
          
          cv.imshow('Interface', frame)

    self.cap.release()
    cv.destroyAllWindows()
    
if __name__ == '__main__':
  iris_extration = Iris()
  iris_extration.Detection()