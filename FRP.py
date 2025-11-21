from keras_facenet import FaceNet
import cv2
import numpy as np
from numpy import linalg as LA
from mtcnn import MTCNN

embedder = FaceNet()
detector = MTCNN()

def preprocess_img(img_path):
    
    img = cv2.imread(img_path)
    
    if img is None:
        raise Exception("Image not found or unreadable!")

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    results = detector.detect_faces(img)
    
    if len(results) == 0:
        raise Exception("No face detected!")

    x, y, w, h = results[0]['box']

    x = max(0, x)
    y = max(0, y)
    w = max(1, w)
    h = max(1, h)

    x2 = min(x + w, img.shape[1])
    y2 = min(y + h, img.shape[0])

    face = img[y:y2, x:x2]
    
    if face.size == 0 or face.shape[0] == 0 or face.shape[1] == 0:
        raise Exception("Face crop is empty! Bounding box out of range!")

    # face = cv2.resize(face, (160, 160))
    # face = face.astype('float32') / 255.0
    
    cv2.imwrite("debug_face.jpg", cv2.cvtColor(face, cv2.COLOR_RGB2BGR))
    
    return np.expand_dims(face, axis=0)


def get_face_embedding(model, img_path):
    
    img = preprocess_img(img_path)
    embedding = embedder.embeddings(img).squeeze()
    
    return embedding

def face_distance(embedding1, embedding2):
    return np.linalg.norm(embedding1 - embedding2)

def compare_faces(embedding1 , embedding2, threshold = 0.5):
    
    distance = face_distance(embedding1, embedding2)

    if distance < threshold:
        print(f"Matched (distance = {distance:.4f})")
    else:
        print(f"Different (distance = {distance:.4f})")

    return distance



embedding1 = get_face_embedding(embedder, '2.jpg')
embedding2 = get_face_embedding(embedder, 'test8.jpg')

distance = compare_faces(embedding1, embedding2)

print(f"Euclidean Distance: {distance}")

test = cv2.imread('data/actor1/2.jpg')
test = cv2.cvtColor(test, cv2.COLOR_BGR2RGB)

results = detector.detect_faces(test)
print(results)



