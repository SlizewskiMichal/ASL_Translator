import tensorflow as tf
import numpy as np


def wczytaj_model_ASL(path):
    model = tf.keras.Sequential([
        tf.keras.layers.Flatten(input_shape=(21, 2, 1)),
        tf.keras.layers.Dense(128, activation='sigmoid'),
        tf.keras.layers.Dense(24, activation='softmax')
    ])
    model.load_weights(path)
    return model


def predykuj_litere(graf,sesja,punkty_kluczowe, model_asl, litera, LITERY):
    with graf.as_default():
        with sesja.as_default():
            lewa_dlon = tf.expand_dims(punkty_kluczowe, axis=-1)
            predykcje_liter = model_asl.predict(lewa_dlon, steps=100)
            if predykcje_liter[0][np.argmax(predykcje_liter[0])] >= 0.9:
                przewidywana_litera = (LITERY[np.argmax(predykcje_liter[0])])
                return przewidywana_litera
            return litera


def wczytaj_graf_ASL(konfiguracja, SCIEZKA_DO_MODELU_ASL):
    graf = tf.get_default_graph()
    sesja = tf.compat.v1.Session(config=konfiguracja)

    with graf.as_default():
        with sesja.as_default():
            model = wczytaj_model_ASL(SCIEZKA_DO_MODELU_ASL)
    return model, graf, sesja
