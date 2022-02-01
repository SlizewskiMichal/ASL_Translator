import simpleaudio as sa
from ASL import predykuj_litere, wczytaj_graf_ASL
from punkty_kluczowe import wczytaj_graf_punkty_kluczowe, predykuj_punkty_kluczowe
from dlonie import wczytaj_graf_dlonie, predykuj_dlonie
# Inicjowanie wszystkich zmiennych
from zmienne import *


def odtworz_dzwiek(sciezka_do_dzwieku):
    try:
        plik_wav = sa.WaveObject.from_wave_file(sciezka_do_dzwieku)
        plik_wav.play()
    except FileNotFoundError:
        pass


if __name__ == '__main__':
    # Wczytywanie potrzebnych modeli, grafów oraz sesji
    graf_detekcja_dloni, sesja_detekcja_dloni = wczytaj_graf_dlonie(SCIEZKA_DO_GRAFU_WYKRYWANIE_DLONI)
    model_punkty_kluczowe, graf_punkty_kluczowe, sesja_punkty_kluczowe = wczytaj_graf_punkty_kluczowe(konfiguracja,
                                                                                                      ROZMIAR_OBRAZU_DO_PREDYKCJI,
                                                                                                      LICZBA_PUNKTOW_KLUCZOWYCH,
                                                                                                      SCIEZKA_DO_MODELU_WYKRYWANIE_PUNKTOW_KLUCZOWYCH)
    model_asl, graf_ASL, sesja_ASL = wczytaj_graf_ASL(konfiguracja, SCIEZKA_DO_MODELU_ASL)

    # Główna pętla programu
    while True:
        # Przechwytywanie obrazów z kamerki
        obraz_przechwycony, obraz = kamerka.read()
        if obraz_przechwycony:
            # Zamiana na BGR
            obraz = cv2.cvtColor(obraz, cv2.COLOR_RGB2BGR)
            # Predykcja polozenia dloni na zdjeciu
            boxy, obraz_z_boxami = predykuj_dlonie(graf_detekcja_dloni, sesja_detekcja_dloni, obraz, szerokosc_zdjecia,
                                                   wysokosc_zdjecia)
            # Inicjacja zmiennych odpowiadających za przechowywanie położeń lewej i prawej dłoni
            lewa_dlon = None
            prawa_dlon = None
            # Wykrycie lewej oraz prawej dłoni
            for box in boxy:
                if int(box[0]) < obraz.shape[1] / 2:
                    lewa_dlon = box
                else:
                    prawa_dlon = box

            # Predykcja położeń punktów kluczowych dla lewej dłoni
            if lewa_dlon is not None:
                lewa_dlon = predykuj_punkty_kluczowe(graf_punkty_kluczowe, sesja_punkty_kluczowe, obrazy_do_predykcji,
                                                     obraz, lewa_dlon, model_punkty_kluczowe,
                                                     ROZMIAR_OBRAZU_DO_PREDYKCJI)

            # Predykcja położeń punktów kluczowych dla prawej dłoni
            if prawa_dlon is not None:
                prawa_dlon = (
                    prawa_dlon,
                    predykuj_punkty_kluczowe(graf_punkty_kluczowe, sesja_punkty_kluczowe, obrazy_do_predykcji,
                                             obraz, prawa_dlon, model_punkty_kluczowe,
                                             ROZMIAR_OBRAZU_DO_PREDYKCJI))

            if lewa_dlon is not None:
                # Predykcja pokazywanej litery
                przewidywana_litera = predykuj_litere(graf_ASL, sesja_ASL, lewa_dlon, model_asl, przewidywana_litera,
                                                      LITERY)
                # Pisanie litery na ekranie
                cv2.putText(obraz_z_boxami, przewidywana_litera,
                            miejsce_na_napis,
                            font,
                            wielkosc_czcionki,
                            kolor_czcionki,
                            grubisc_czcionki,
                            typ_lini)
            # Napis jeśli prawa dłoń wykona gest zamknięcia
            if prawa_dlon is not None:

                y = [i[1] for i in prawa_dlon[-1][0]]
                x = [i[0] for i in prawa_dlon[-1][0]]

                # Jeśli dłoń była otwarta ale teraz się zamknęła, jeśli nie to dłoń otwarta
                if x.index(min(x)) not in [8, 12, 16, 20, 7, 11, 15, 19] and dlon_otwarta:
                    cv2.putText(obraz_z_boxami, NAPIS,
                                (int(prawa_dlon[0][0]), int(prawa_dlon[0][3] - 25)),
                                font,
                                1,
                                (0, 200, 0),
                                1,
                                typ_lini)
                    # Odtwrzanie dźwięku za pomocą modułu simpleaudio
                    odtworz_dzwiek('dzwieki/' + przewidywana_litera.capitalize() + '.wav')
                    # dłoń jest teraz zamknięta
                    dlon_otwarta = False
                else:
                    dlon_otwarta = True

            # Konwersja obrazu spowrotem na RGB
            obraz_z_boxami = cv2.cvtColor(obraz_z_boxami, cv2.COLOR_BGR2RGB)

            # Pokazywanie obrazu
            cv2.imshow('ASL Translator', obraz_z_boxami)
            if cv2.waitKey(25) & 0xFF == ord('q'):
                cv2.destroyAllWindows()
                break
