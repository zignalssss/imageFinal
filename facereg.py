import cv2
import face_recognition
import os

# โฟลเดอร์ที่เก็บภาพของบุคคลที่รู้จัก
known_faces_dir = "/Users/wiritipon/Desktop/imageProject/known_faces"
known_encodings = []
known_names = []

# โหลดภาพและสร้างค่า encoding สำหรับบุคคลแต่ละคน
for filename in os.listdir(known_faces_dir):
    if filename.endswith(('.jpg', '.jpeg', '.png')):
        image_path = os.path.join(known_faces_dir, filename)
        
        name = os.path.splitext(filename)[0]
        
        image = face_recognition.load_image_file(image_path)
        encodings = face_recognition.face_encodings(image)
        if encodings:
            known_encodings.append(encodings[0])
            known_names.append(name)

# เปิดการเชื่อมต่อกับเว็บแคม
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: ไม่สามารถเปิดเว็บแคมได้.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: ไม่สามารถจับภาพได้.")
        break

    # แปลงเฟรมจาก BGR เป็น RGB และให้เป็น 8-bit
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).astype('uint8')

    # ตรวจจับตำแหน่งใบหน้าและเข้ารหัสใบหน้าในเฟรม
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        matches = face_recognition.compare_faces(known_encodings, face_encoding)
        name = "Unknown"

        if True in matches:
            matched_idx = matches.index(True)
            name = known_names[matched_idx]

        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    cv2.imshow('Real-Time Face Recognition', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
