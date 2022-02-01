import numpy as np
import copy
import cv2
import tensorflow as tf


def wykryj_dlonie(obraz, graf_dlonie, sesja):
    obraz_tensor = graf_dlonie.get_tensor_by_name('image_tensor:0')
    boxy = graf_dlonie.get_tensor_by_name(
        'detection_boxes:0')
    predykcje = graf_dlonie.get_tensor_by_name(
        'detection_scores:0')
    klasy = graf_dlonie.get_tensor_by_name(
        'detection_classes:0')
    liczba_detekcji = graf_dlonie.get_tensor_by_name(
        'num_detections:0')

    obraz_roszerzony = np.expand_dims(obraz, axis=0)

    (boxy, predykcje, klasy, liczba_detekcji) = sesja.run(
        [boxy, predykcje,
         klasy, liczba_detekcji],
        feed_dict={obraz_tensor: obraz_roszerzony})
    return np.squeeze(boxy), np.squeeze(predykcje)


def rysuj_boxy(prog_wykrycia, predykcje, boxy, szerokosc_zdjecia, wysokosc_zdjecia, obraz):
    dane_boxow = []
    obraz_z_boxami = copy.copy(obraz)
    for i in range(2):

        if predykcje[i] > prog_wykrycia:
            (lewa, prawa, gora, dol) = (boxy[i][1] * szerokosc_zdjecia, boxy[i][3] * szerokosc_zdjecia,
                                        boxy[i][0] * wysokosc_zdjecia, boxy[i][2] * wysokosc_zdjecia)
            miny, maxy, minx, maxx = int(gora), int(dol), int(lewa), int(prawa)
            minx = minx - 20 if minx - 20 > 0 else minx
            miny = miny - 20 if miny - 20 > 0 else miny
            maxx = maxx + 20 if maxx + 20 < obraz.shape[1] else maxx
            maxy = maxy + 20 if maxy + 20 < obraz.shape[0] else maxy

            p1 = (minx, maxy)
            p2 = (maxx, miny)
            cv2.rectangle(obraz_z_boxami, p1, p2, (77, 255, 9), 3, 1)

            dane_boxow.append((lewa, prawa, gora, dol, obraz_z_boxami))
    return dane_boxow, obraz_z_boxami

def wczytaj_graf_dlonie(SCIEZKA_DO_GRAFU_WYKRYWANIE_DLONI):
    graf_detekcja_dloni = tf.Graph()
    with graf_detekcja_dloni.as_default():
        graf_dlonie = tf.GraphDef()
        with tf.compat.v1.gfile.GFile(SCIEZKA_DO_GRAFU_WYKRYWANIE_DLONI, 'rb') as gfile:
            graf = gfile.read()
            graf_dlonie.ParseFromString(graf)
            tf.import_graph_def(graf_dlonie, name='')
    sesja = tf.compat.v1.Session(graph=graf_detekcja_dloni)
    return graf_detekcja_dloni,sesja

def predykuj_dlonie(graf,sesja,obraz,szerokosc_zdjecia,wysokosc_zdjecia):
    with graf.as_default():
        with sesja.as_default():
            boxy, predykcje = wykryj_dlonie(obraz, graf, sesja)

            boxy, obraz_z_boxami = rysuj_boxy(0.3, predykcje, boxy, szerokosc_zdjecia, wysokosc_zdjecia, obraz)

            return boxy,obraz_z_boxami
