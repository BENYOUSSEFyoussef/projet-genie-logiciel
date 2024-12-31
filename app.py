import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Flatten, Dense
import cv2
import os
from flask import Flask, request, render_template
from datetime import date, datetime
import numpy as np00
import pandas as pd
app = Flask(__name__)
nimgs = 10

imgBackground=cv2.imread("background.png")

datetoday = date.today().strftime("%m_%d_%y")
datetoday2 = date.today().strftime("%d-%B-%Y")
face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
if not os.path.isdir('Attendance'):
    os.makedirs('Attendance')
if not os.path.isdir('static'):
    os.makedirs('static')
if not os.path.isdir('static/faces'):
    os.makedirs('static/faces')
if f'Attendance-{datetoday}.csv' not in os.listdir('Attendance'):
    with open(f'Attendance/Attendance-{datetoday}.csv', 'w') as f:
        f.write('Name,Roll,Time')

def totalreg():
    return len(os.listdir('static/faces'))

def extract_faces(img):
    try:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        face_points = face_detector.detectMultiScale(gray, 1.2, 5, minSize=(20, 20))
        return face_points
    except Exception as e:
        print(f"Erreur d'extraction de visage : {e}")
        return []

def identify_face(facearray, confidence_threshold=0.5):
    # Charger le modèle
    model = load_model('static/modele.keras')
    
    # Prétraiter l'image du visage
    facearray = cv2.resize(facearray, (224, 224))
    facearray = facearray / 255.0  # Normalisation des pixels entre 0 et 1
    facearray = np.expand_dims(facearray, axis=0)  # Ajouter la dimension du batch
    
    # Effectuer la prédiction
    prediction = model.predict(facearray)
    
    # Obtenir l'indice de la classe avec la plus forte probabilité
    predicted_class = np.argmax(prediction)
    confidence = prediction[0][predicted_class]  # La probabilité associée à la classe prédite
    
    # Vérifier si la confiance dépasse le seuil
    if confidence >= confidence_threshold:
        userlist = os.listdir('static/faces')
        user_names = [user for user in userlist]  # Utiliser directement le nom du dossier
        return user_names[predicted_class], confidence
    else:
        return "Inconnu", confidence

def train_model():
    faces = []
    labels = []
    userlist = os.listdir('faces/dataset/train')
    for user in userlist:
        for imgname in os.listdir(f'faces/dataset/train/{user}'):
            img = cv2.imread(f'faces/dataset/train/{user}/{imgname}')
            resized_face = cv2.resize(img, (224, 224))  # Dimension compatible avec VGG16
            faces.append(resized_face / 255.0)  # Normalisation des pixels entre 0 et 1
            labels.append(user)
    
    faces = np.array(faces)
    labels = np.array(pd.factorize(labels)[0])  # Conversion des labels en indices numériques
    
    # Charger le modèle VGG16 pré-entraîné sans la couche de classification
    vgg_base = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    
    # Personnaliser le modèle
    x = Flatten()(vgg_base.output)
    x = Dense(128, activation='relu')(x)
    output = Dense(len(np.unique(labels)), activation='softmax')(x)  # Nombre de classes = nombre d'utilisateurs
    model = Model(inputs=vgg_base.input, outputs=output)
    
    # Geler les couches de VGG16
    for layer in vgg_base.layers:
        layer.trainable = False
    
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001), 
                  loss='sparse_categorical_crossentropy', 
                  metrics=['accuracy'])
    
    # Entraîner le modèle
    model.fit(faces, labels, epochs=10, batch_size=16)  # Vous pouvez augmenter les époques ou le batch size
    
    model.save('static/modele.keras')
def extract_attendance():
    df = pd.read_csv(f'Attendance/Attendance-{datetoday}.csv')
    names = df['Name']
    rolls = df['Roll']
    times = df['Time']
    l = len(df)
    return names, rolls, times, l
def add_attendance(name):
    username = name  # Juste le nom
    current_time = datetime.now().strftime("%H:%M:%S")
    
    df = pd.read_csv(f'Attendance/Attendance-{datetoday}.csv')
    if username not in list(df['Name']):
        with open(f'Attendance/Attendance-{datetoday}.csv', 'a') as f:
            f.write(f'\n{username},N/A,{current_time}')  # N/A au lieu d'un ID

def getallusers():
    userlist = os.listdir('faces/dataset/train')
    names = []
    rolls = []
    l = len(userlist)

    for i in userlist:
        names.append(i)  # Le nom de la classe est le nom du dossier
        rolls.append("")  # Aucun ID disponible

    return userlist, names, rolls, l
@app.route('/')
def home():
    names, rolls, times, l = extract_attendance()
    return render_template('home.html', names=names, rolls=rolls, times=times, l=l, totalreg=totalreg(), datetoday2=datetoday2)

@app.route('/start', methods=['GET'])
def start():
    names, rolls, times, l = extract_attendance()

    if 'modele.keras' not in os.listdir('static'):
        return render_template('home.html', names=names, rolls=rolls, times=times, l=l, totalreg=totalreg(), datetoday2=datetoday2, mess='Aucun modèle VGG16 entraîné. Veuillez ajouter un utilisateur.')
    
    cap = cv2.VideoCapture(0)
    while True:
        _, frame = cap.read()
        faces = extract_faces(frame)
        for (x, y, w, h) in faces:
            face = frame[y:y+h, x:x+w]
            
            # Appeler la fonction d'identification avec le seuil de confiance
            identified_person, confidence = identify_face(face)
            
            # Ajouter l'attendance si la personne est identifiée avec une confiance suffisante
            if identified_person != "Inconnu":
                add_attendance(identified_person)
            
            # Afficher le nom de la personne et la confiance
            cv2.putText(frame, f'{identified_person} ({confidence:.2f})', (x, y-15), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        cv2.imshow('Attendance', frame)
        if cv2.waitKey(1) == 27:  # Quitter lorsque la touche "Esc" est pressée
            break
    
    cap.release()
    cv2.destroyAllWindows()
    names, rolls, times, l = extract_attendance()
    return render_template('home.html', names=names, rolls=rolls, times=times, l=l, totalreg=totalreg(), datetoday2=datetoday2)

@app.route('/add', methods=['GET', 'POST'])
def add():
    newusername = request.form['newusername']
    
    userimagefolder = f'static/faces/{newusername}'  # Créer un dossier uniquement avec le nom
    
    if not os.path.isdir(userimagefolder):
        os.makedirs(userimagefolder)
    
    i, j = 0, 0
    cap = cv2.VideoCapture(0)
    while True:
        _, frame = cap.read()
        faces = extract_faces(frame)
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 20), 2)
            cv2.putText(frame, f'Images Captured: {i}/{nimgs}', (30, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 20), 2, cv2.LINE_AA)
            
            if j % 5 == 0:
                name = f"{i}.jpg"  # Sauvegarder l'image sans le nom et l'ID
                face_img = frame[y:y+h, x:x+w]
                resized_face = cv2.resize(face_img, (224, 224))
                cv2.imwrite(userimagefolder + '/' + name, resized_face)
                i += 1
            
            j += 1
        if j == nimgs * 5:
            break
        cv2.imshow('Adding new User', frame)
        if cv2.waitKey(1) == 27:
            break
    
    cap.release()
    cv2.destroyAllWindows()
    
    train_model()
    names, rolls, times, l = extract_attendance()
    return render_template('home.html', names=names, rolls=rolls, times=times, l=l, totalreg=totalreg(), datetoday2=datetoday2)
if __name__ == '__main__':
    app.run(debug=True)
