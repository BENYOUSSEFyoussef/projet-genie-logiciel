import os
import shutil
import random

# 📂 Chemin principal du dataset
base_dir = 'dataset'  # Répertoire contenant les images des personnes
train_dir = 'dataset/train/'  # Répertoire d'entraînement
test_dir = 'dataset/test/'  # Répertoire de test

# 📊 Pourcentage de division (80% entraînement, 20% test)
split_ratio = 0.8  # 80% des images pour le train, 20% pour le test

# 🛠️ Création des dossiers de train et test
if not os.path.exists(train_dir):
    os.makedirs(train_dir)

if not os.path.exists(test_dir):
    os.makedirs(test_dir)

# 🔄 Parcourir chaque personne dans le dataset
for person_name in os.listdir(base_dir):
    person_path = os.path.join(base_dir, person_name)
    
    # Ignorer les dossiers de train et test (pour éviter de les traiter)
    if person_name in ['train', 'test']:
        continue
    
    # Vérifiez si c'est un répertoire
    if os.path.isdir(person_path):
        print(f"Traitement de la personne : {person_name}...")

        # Création des dossiers de la personne dans le train et le test
        train_person_path = os.path.join(train_dir, person_name)
        test_person_path = os.path.join(test_dir, person_name)

        if not os.path.exists(train_person_path):
            os.makedirs(train_person_path)
        if not os.path.exists(test_person_path):
            os.makedirs(test_person_path)

        # 📂 Récupérer toutes les images de la personne
        images = os.listdir(person_path)
        images = [img for img in images if img.endswith(('.jpg', '.jpeg', '.png'))]  # Filtrer uniquement les images
        
        # Mélanger les images au hasard pour avoir un split aléatoire
        random.shuffle(images)

        # 📊 Calcul du nombre d'images pour le train et le test
        split_point = int(len(images) * split_ratio)
        
        # 📁 Images d'entraînement
        train_images = images[:split_point]
        
        # 📁 Images de test
        test_images = images[split_point:]

        # 📦 Déplacement des images d'entraînement
        for img in train_images:
            source_path = os.path.join(person_path, img)
            destination_path = os.path.join(train_person_path, img)
            shutil.copy(source_path, destination_path)  # Copier les images (changez en shutil.move si vous voulez les déplacer)

        # 📦 Déplacement des images de test
        for img in test_images:
            source_path = os.path.join(person_path, img)
            destination_path = os.path.join(test_person_path, img)
            shutil.copy(source_path, destination_path)  # Copier les images (changez en shutil.move si vous voulez les déplacer)

        print(f"  - {len(train_images)} images pour l'entraînement.")
        print(f"  - {len(test_images)} images pour le test.\n")
