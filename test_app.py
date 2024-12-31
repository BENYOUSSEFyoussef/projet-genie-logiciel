import unittest
import os
from datetime import datetime
from app import totalreg, extract_attendance, train_model, add_attendance, getallusers

class TestFaceRecognitionApp(unittest.TestCase):

    def setUp(self):
        """Avant chaque test, nous nettoyons les répertoires et préparons l'environnement."""
        self.attendance_folder = 'Attendance'
        self.faces_folder = 'static/faces'
        self.model_path = 'static/face_recognition_model.pkl'
        self.datetoday = datetime.today().strftime('%Y-%m-%d')

        # Nettoyage des répertoires
        if not os.path.exists(self.attendance_folder):
            os.makedirs(self.attendance_folder)
        if not os.path.exists(self.faces_folder):
            os.makedirs(self.faces_folder)

        # Création d'un fichier d'attendance vide si nécessaire
        if f'Attendance-{self.datetoday}.csv' not in os.listdir(self.attendance_folder):
            with open(f'{self.attendance_folder}/Attendance-{self.datetoday}.csv', 'w') as f:
                f.write('Name,Roll,Time')

    def test_totalreg(self):
        """Test de la fonction totalreg() qui renvoie le nombre total d'utilisateurs enregistrés."""
        self.assertEqual(totalreg(), 1)

    def test_add_attendance(self):
        """Test d'ajout de présence."""
        # Ajouter un exemple de présence
        add_attendance('youssef_1')
        names, rolls, times, l = extract_attendance()
        self.assertIn('youssef', names)
        self.assertIn('1', rolls)

    def test_train_model(self):
        """Test de l'entraînement du modèle de reconnaissance faciale."""
        # Entraînement du modèle avec les images présentes
        train_model()
        self.assertTrue(os.path.exists(self.model_path))

    def test_getallusers(self):
        """Test de la fonction getallusers() qui récupère les utilisateurs enregistrés."""
        userlist, names, rolls, l = getallusers()
        self.assertEqual(l, 1)  # Devrait être 0 si aucun utilisateur n'a été ajouté

    def test_attendance_file_generation(self):
        """Test de la génération du fichier d'attendance CSV."""
        add_attendance('Jane_456')
        names, rolls, times, l = extract_attendance()
        self.assertEqual(l, 1)  # Après ajout d'une présence, il devrait y avoir 1 entrée
        self.assertTrue(f'Attendance-{self.datetoday}.csv' in os.listdir(self.attendance_folder))

    def tearDown(self):
        """Après chaque test, nettoyer l'environnement."""
        # Supprimer les fichiers de test pour garder un environnement propre
        if os.path.exists(self.attendance_folder):
            for file in os.listdir(self.attendance_folder):
                file_path = os.path.join(self.attendance_folder, file)
                if os.path.isfile(file_path):
                    os.remove(file_path)
        if os.path.exists(self.faces_folder):
            for file in os.listdir(self.faces_folder):
                file_path = os.path.join(self.faces_folder, file)
                if os.path.isfile(file_path):
                    os.remove(file_path)

if __name__ == '__main__':
    unittest.main()
