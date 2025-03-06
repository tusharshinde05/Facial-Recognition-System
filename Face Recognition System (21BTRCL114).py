import numpy as np
import joblib
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from mtcnn import MTCNN
from keras_facenet import FaceNet
import cv2
import os
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk


def get_embedding(face_img, embedder):
    face_img = face_img.astype('float32')
    face_img = np.expand_dims(face_img, axis=0)
    yhat = embedder.embeddings(face_img)
    return yhat[0]


def extract_faces(directory, detector, embedder):
    X, Y = [], []
    for sub_dir in os.listdir(directory):
        path = os.path.join(directory, sub_dir)
        if not os.path.isdir(path):
            continue
        
        for filename in os.listdir(path):
            try:
                file_path = os.path.join(path, filename)
                image = cv2.imread(file_path)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                faces = detector.detect_faces(image)
                if faces:
                    x, y, w, h = faces[0]['box']
                    face_img = image[y:y+h, x:x+w]
                    face_img = cv2.resize(face_img, (160, 160))
                    embedding = get_embedding(face_img, embedder)
                    X.append(embedding)
                    Y.append(sub_dir)
            except Exception as e:
                print(f"Skipping {filename}: {e}")
    return np.asarray(X), np.asarray(Y)


# Load dataset
dataset_directory = "dataset-20240224T123354Z-001/dataset"
detector = MTCNN()
embedder = FaceNet()
X, Y = extract_faces(dataset_directory, detector, embedder)

# Split dataset
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=17, shuffle=True)

# Train SVM model
model = SVC(kernel='linear', probability=True)
model.fit(X_train, Y_train)

# Save model
model_path = "svm_model_160x160.pkl"
joblib.dump(model, model_path)
print(f"Model saved to {model_path}")


def recognize_faces(frame, detector, svc_model, embedder, labels):
    faces = detector.detect_faces(frame)
    for face in faces:
        bounding_box = face['box']
        x, y, w, h = bounding_box
        face_img = frame[y:y+h, x:x+w]
        face_img = cv2.resize(face_img, (160, 160))
        embedding = get_embedding(face_img, embedder)
        label = str(svc_model.predict([embedding])[0])
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)
    return frame


def select_image():
    file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg;*.png;*.jpeg")])
    if file_path:
        process_image(file_path)


def process_image(image_path):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    faces = detector.detect_faces(image)
    
    if faces:
        for face in faces:
            x, y, w, h = face['box']
            face_img = image[y:y+h, x:x+w]
            face_img = cv2.resize(face_img, (160, 160))
            embedding = get_embedding(face_img, embedder)
            label = str(svc_model.predict([embedding])[0])
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(image, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)
    else:
        print("No face detected in the image.")
        
    display_image(image)


def display_image(image_array):
    image = Image.fromarray(image_array)
    image = ImageTk.PhotoImage(image)
    panel.config(image=image)
    panel.image = image


def process_video():
    cap = cv2.VideoCapture('WIN_20250223_21_34_29_Pro.mp4')
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter('output.avi', fourcc, 20.0, (int(cap.get(3)), int(cap.get(4))))
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        processed_frame = recognize_faces(frame, detector, svc_model, embedder, labels)
        cv2.imshow('Video Face Recognition', processed_frame)
        out.write(processed_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    out.release()
    cv2.destroyAllWindows()


# Load pre-trained model
svc_model_path = 'svm_model_160x160.pkl'
svc_model = joblib.load(svc_model_path)
embedder = FaceNet()
detector = MTCNN()
labels = ['jenna_ortega', 'robert_downey', 'sardor_abdirayimov', 'taylor_swift', 'Tushar']

# Create GUI
root = tk.Tk()
root.title("Face Recognition System")

btn_img = tk.Button(root, text="Select Image", command=select_image)
btn_img.pack()

btn_vid = tk.Button(root, text="Process Video", command=process_video)
btn_vid.pack()

panel = tk.Label(root)
panel.pack()

root.mainloop()