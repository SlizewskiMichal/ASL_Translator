import tensorflow as tf
import numpy as np
import cv2

#Sciezki do modeli
SCIEZKA_DO_GRAFU_WYKRYWANIE_DLONI = 'modele/frozen_inference_graph.pb'
SCIEZKA_DO_MODELU_WYKRYWANIE_PUNKTOW_KLUCZOWYCH = 'modele/model_keypoints_ASL.h5'
SCIEZKA_DO_MODELU_ASL = 'modele/x.ckpt'
#Stale Etykiety
LICZBA_PUNKTOW_KLUCZOWYCH = 21
ROZMIAR_OBRAZU_DO_PREDYKCJI = 224
NAPIS = 'POWIEDZ'
LITERY = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U',
          'V', 'W', 'X', 'Y']
konfiguracja = tf.ConfigProto(
    device_count={'CPU': 2},
    intra_op_parallelism_threads=10,
    allow_soft_placement=True
)

font = cv2.FONT_HERSHEY_SIMPLEX
miejsce_na_napis = (10, 150)
wielkosc_czcionki = 6
kolor_czcionki = (0, 0, 0)
grubisc_czcionki = 4
typ_lini = 5
kamerka = cv2.VideoCapture(0)
kamerka.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
kamerka.set(cv2.CAP_PROP_FRAME_HEIGHT, 180)
szerokosc_zdjecia, wysokosc_zdjecia = (kamerka.get(3), kamerka.get(4))
#poczatkowa inicjalizacja zmiennych
obrazy_do_predykcji = np.empty((1, ROZMIAR_OBRAZU_DO_PREDYKCJI, ROZMIAR_OBRAZU_DO_PREDYKCJI, 3), dtype="int")
dlon_otwarta = True
przewidywana_litera = ' '

