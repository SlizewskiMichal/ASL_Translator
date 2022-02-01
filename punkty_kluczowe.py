from tensorflow import keras
from tensorflow.keras import layers
import cv2
import tensorflow as tf


def wczytaj_model_punkty_kluczowe(ROZMIAR_OBRAZU_DO_PREDYKCJI, LICZBA_PUNKTOW_KLUCZOWYCH,
                                  SCIEZKA_DO_MODELU_WYKRYWANIE_PUNKTOW_KLUCZOWYCH):
    mobilenet = keras.applications.MobileNetV2(
        weights="imagenet", include_top=False, input_shape=(ROZMIAR_OBRAZU_DO_PREDYKCJI, ROZMIAR_OBRAZU_DO_PREDYKCJI, 3)
    )
    mobilenet.trainable = True

    wejscia = layers.Input((ROZMIAR_OBRAZU_DO_PREDYKCJI, ROZMIAR_OBRAZU_DO_PREDYKCJI, 3))
    model = keras.applications.mobilenet_v2.preprocess_input(wejscia)
    model = mobilenet(model)
    model = layers.Dropout(0.3)(model)
    model = layers.SeparableConv2D(
        LICZBA_PUNKTOW_KLUCZOWYCH * 2, kernel_size=5, strides=1, activation="relu"
    )(model)
    model = layers.SeparableConv2D(
        2 * LICZBA_PUNKTOW_KLUCZOWYCH, kernel_size=3, strides=1, activation="sigmoid"
    )(model)

    model = keras.Model(wejscia, model, name="keypoint_detector")
    model.load_weights(SCIEZKA_DO_MODELU_WYKRYWANIE_PUNKTOW_KLUCZOWYCH)
    return model


def predykuj_punkty_kluczowe(graf,sesja,obraz_do_predykcji, obraz, box, model, ROZMIAR_OBRAZU_DO_PREDYKCJI):
    with graf.as_default():
        with sesja.as_default():
            miny, maxy, minx, maxx = int(box[2]), int(box[3]), int(box[0]), int(box[1])

            kopia_obrazu = obraz[miny:maxy, minx:maxx, :]
            kopia_obrazu = cv2.resize(kopia_obrazu, (ROZMIAR_OBRAZU_DO_PREDYKCJI, ROZMIAR_OBRAZU_DO_PREDYKCJI))
            obraz_do_predykcji[0] = kopia_obrazu
            return model.predict([obraz_do_predykcji]).reshape(-1, 21, 2) * ROZMIAR_OBRAZU_DO_PREDYKCJI


def wczytaj_graf_punkty_kluczowe(konfiguracja, ROZMIAR_OBRAZU_DO_PREDYKCJI,
                                 LICZBA_PUNKTOW_KLUCZOWYCH,
                                 SCIEZKA_DO_MODELU_WYKRYWANIE_PUNKTOW_KLUCZOWYCH):
    sesja = tf.compat.v1.Session(config=konfiguracja)
    graf_punkty_kluczowe = tf.get_default_graph()
    with graf_punkty_kluczowe.as_default():
        with sesja.as_default():
            model = wczytaj_model_punkty_kluczowe(ROZMIAR_OBRAZU_DO_PREDYKCJI,
                                                  LICZBA_PUNKTOW_KLUCZOWYCH,
                                                  SCIEZKA_DO_MODELU_WYKRYWANIE_PUNKTOW_KLUCZOWYCH)
    return model, graf_punkty_kluczowe, sesja
