import tkinter as tk
from tkinter import filedialog, messagebox
import numpy as np
import librosa
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq

def lire_wav(fichier):
    y, sr = librosa.load(fichier)
    return y, sr

def tracer_wave(data, framerate):
    duree = len(data) / framerate
    temps = np.linspace(0, duree, num=len(data))
    
    plt.figure(figsize=(12, 6))
    plt.plot(temps, data)
    plt.title("Forme d'onde du fichier audio")
    plt.xlabel("Temps (secondes)")
    plt.ylabel("Amplitude")
    plt.grid(True)
    plt.show()

def determiner_frequence(data, framerate):
    N = len(data)
    yf = fft(data)
    xf = fftfreq(N, 1 / framerate)
    
    xf = xf[:N//2]
    yf = np.abs(yf[:N//2])
    
    idx_frequence_dominante = np.argmax(yf)
    frequence_dominante = xf[idx_frequence_dominante]
    
    return frequence_dominante

def calculer_tristimulus(y, sr):
    D = librosa.stft(y)
    magnitudes = np.abs(D)
    
    f0, voiced_flag, _ = librosa.pyin(y, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'))
    f0 = f0[voiced_flag]
    
    t1, t2, t3 = [], [], []
    
    for f in f0:
        if np.isnan(f):
            t1.append(0)
            t2.append(0)
            t3.append(0)
            continue
        
        harmonics = np.arange(1, 11) * f
        harmonic_mags = [np.mean(magnitudes[int(h/sr*magnitudes.shape[0])] if h < sr/2 else 0) for h in harmonics]
        
        total_energy = sum(harmonic_mags)
        if total_energy == 0:
            t1.append(0)
            t2.append(0)
            t3.append(0)
        else:
            t1.append(harmonic_mags[0] / total_energy)
            t2.append(sum(harmonic_mags[1:4]) / total_energy)
            t3.append(sum(harmonic_mags[4:]) / total_energy)
    
    return np.mean(t1), np.mean(t2), np.mean(t3)

def calculer_descripteurs(y, sr):
    descripteurs = {}
    
    descripteurs['Centroïde spectral'] = (np.mean(librosa.feature.spectral_centroid(y=y, sr=sr)), 'Hz', """Centroïde spectral :
- C'est la "moyenne" des fréquences présentes dans le signal, pondérée par leurs amplitudes.
- Il représente le "centre de gravité" du spectre."""
    )
    etalement_spectral_explication = """Étalement spectral :
- Mesure à quel point les fréquences du signal sont dispersées autour de ce centroïde.
- Une valeur faible indique que l'énergie est concentrée autour du centroïde.
- Une valeur élevée indique que l'énergie est répartie sur une large gamme de fréquences.

Concrètement, l'étalement spectral nous renseigne sur :

1. La "largeur" du spectre :
   - Un son avec un étalement faible aura un spectre étroit, concentré autour d'une fréquence principale.
   - Un son avec un étalement élevé aura un spectre large, avec de l'énergie répartie sur de nombreuses fréquences.

2. La "pureté" ou la "complexité" du son :
   - Un son pur (comme une sinusoïde) aura un étalement très faible.
   - Un son complexe ou bruité aura un étalement plus important.

3. Le caractère tonal ou bruité du son :
   - Les sons tonaux (comme une note de piano) ont généralement un étalement plus faible.
   - Les sons bruités (comme un bruit blanc) ont un étalement plus élevé.

Par exemple :
- Une flûte jouant une note aura généralement un étalement spectral plus faible qu'une cymbale frappée.
- Un murmure aura un étalement spectral plus faible qu'un cri.

Exemple 1 : Note de piano vs Bruit blanc
Note de piano (son tonal) :
Centroïde spectral : 1000 Hz
Étalement spectral : 200 Hz
Bruit blanc (son bruité) :
Centroïde spectral : 5000 Hz
Étalement spectral : 4000 Hz
Dans cet exemple, la note de piano a un étalement spectral relativement faible (200 Hz) par rapport à son centroïde (1000 Hz), ce qui indique que l'énergie est concentrée autour de la fréquence fondamentale et de quelques harmoniques. En revanche, le bruit blanc a un étalement spectral beaucoup plus large (4000 Hz), reflétant une distribution d'énergie plus uniforme sur une large gamme de fréquences.
Exemple 2 : Flûte vs Cymbale
Flûte jouant une note :
Centroïde spectral : 800 Hz
Étalement spectral : 150 Hz
Cymbale frappée :
Centroïde spectral : 3000 Hz
Étalement spectral : 2500 Hz
Ici, la flûte, qui produit un son plus pur, a un étalement spectral relativement faible (150 Hz), indiquant que la plupart de l'énergie est concentrée autour de la fréquence fondamentale. La cymbale, en revanche, a un étalement spectral beaucoup plus large (2500 Hz), ce qui reflète la nature complexe et riche en harmoniques de son son
    """
    descripteurs['Étalement spectral'] = (np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr)), 'Hz', etalement_spectral_explication)
    
    descripteurs['Flux spectral'] = (np.mean(librosa.onset.onset_strength(y=y, sr=sr)), '', """Mesure relative du changement du spectre au fil du temps
Définition : Le flux spectral quantifie le taux de changement du spectre de fréquences d'un signal audio d'un instant à l'autre.
                                     
Calcul : Il est généralement calculé en mesurant la différence entre les spectres de magnitude de trames successives du signal.
                                     
Interprétation :
Une valeur élevée de flux spectral indique des changements rapides ou importants dans le contenu fréquentiel du son.
Une valeur faible suggère un contenu spectral plus stable ou des changements plus progressifs.

Applications :
Détection des transitions dans un signal audio (par exemple, changements de notes ou d'accords en musique)
Caractérisation de la dynamique spectrale d'un son
Aide à la segmentation audio ou à la détection d'événements sonores

Unité : Le flux spectral est généralement sans unité, car il représente une mesure relative de changement.Exemples quantitatifs du flux spectral :
                                     
Note de piano soutenue :
Début de la note (attaque) : 0.8
Milieu de la note (sustain) : 0.05
Fin de la note (relâchement) : 0.3
Explication : Le flux spectral est élevé au début en raison du changement soudain lors de l'attaque, très faible pendant la tenue de la note, et augmente légèrement lors du relâchement.

Morceau de musique orchestrale :
Passage calme : 0.1 - 0.2
Crescendo : 0.3 - 0.5
Tutti (tous les instruments jouent) : 0.4 - 0.6
Explication : Le flux spectral augmente avec la complexité et l'intensité de la musique.
                                     
Parole :
Voyelles soutenues : 0.05 - 0.1
Consonnes : 0.3 - 0.7
Transition entre phonèmes : 0.2 - 0.5
Explication : Les consonnes et les transitions entre sons produisent des changements spectraux plus importants que les voyelles soutenues.
                                     
Bruit blanc constant :
Valeur moyenne : 0.02 - 0.05
Explication : Le bruit blanc a un spectre relativement constant, donc le flux spectral est très faible.
                                     
Batterie :
Coup de grosse caisse isolé : 0.9 (au moment de l'impact)
Roulement de caisse claire : 0.6 - 0.8
Cymbale crash : 0.95 (impact initial), puis décroissance à 0.1 - 0.2
Explication : Les percussions produisent des changements spectraux rapides et importants, particulièrement au moment de l'impact.
                                     
Transition entre deux accords de guitare :
Moment du changement d'accord : 0.7 - 0.9
Juste après le changement : 0.3 - 0.5
Accord tenu : 0.05 - 0.1
Explication : Le flux spectral est élevé lors du changement d'accord, puis diminue rapidement une fois l'accord établi.
                                     """
    )
    
    rolloff_percentage = 0.85
    descripteurs['Rolloff spectral'] = (np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr, roll_percent=rolloff_percentage)), 'Hz', f"""Fréquence en dessous de laquelle se trouve {rolloff_percentage*100}% de l\'énergie spectrale
     
Calcul de l'Énergie Spectrale pour le Rolloff Spectral :

1. Transformée de Fourier :
   - On commence par appliquer la Transformée de Fourier au signal audio pour obtenir son spectre de fréquences. 
   - La Transformée de Fourier décompose le signal en une somme de sinusoïdes de différentes fréquences, avec des amplitudes et des phases spécifiques.
   - Si x(t) est le signal temporel, sa Transformée de Fourier X(f) est donnée par :
     X(f) = ∫ x(t) * exp(-j * 2 * pi * f * t) dt

2. Spectre de Puissance :
   - Le spectre de puissance est obtenu en prenant le carré du module de la Transformée de Fourier. Cela nous donne la densité spectrale de puissance (DSP) ou la densité spectrale d'énergie (DSE) si on ne normalise pas par le temps.
   - Pour un signal à énergie finie, la densité spectrale d'énergie Γ_x(f) est donnée par :
     Γ_x(f) = |X(f)|^2
   - Cette densité spectrale représente la quantité d'énergie contenue dans chaque bande de fréquence.

3. Intégration de l'Énergie Spectrale :
   - L'énergie totale du signal dans le domaine fréquentiel est obtenue en intégrant la densité spectrale d'énergie sur toutes les fréquences :
     E_x = ∫ Γ_x(f) df

Utilisation dans le Rolloff Spectral :

Le rolloff spectral est une mesure qui indique la fréquence en dessous de laquelle se trouve un certain pourcentage de l'énergie spectrale totale. Par défaut, ce pourcentage est souvent fixé à 85%.

1. Calcul du Rolloff Spectral :
   - On calcule l'énergie cumulée du spectre en intégrant la densité spectrale d'énergie jusqu'à une certaine fréquence f_r.
   - Le rolloff spectral f_r est la fréquence telle que :
     ∫_0^f_r Γ_x(f) df = 0.85 * E_x
   - Cela signifie que 85% de l'énergie totale du signal est contenue dans les fréquences inférieures ou égales à f_r.
"""                                   
                                        
                                        )
    
    descripteurs['MFCC'] = (np.mean(librosa.feature.mfcc(y=y, sr=sr), axis=1), '', """Représentation compacte du spectre sonore, inspirée de la perception humaine. Utile pour caractériser le timbre.
                            
MFCC (Mel Frequency Cepstral Coefficients) :

1. Définition :
   - Les MFCC sont des coefficients qui représentent le spectre sonore de manière compacte et perceptuellement pertinente.
   - Ils sont largement utilisés dans la reconnaissance vocale et l'analyse musicale.

2. Calcul :
   - On commence par appliquer une Transformée de Fourier discrète (DFT) au signal audio pour obtenir son spectre de fréquences.
   - Le spectre de puissance est ensuite filtré par un banc de filtres sur l'échelle de Mel, qui imite la perception humaine des fréquences.
   - On calcule le logarithme de l'énergie dans chaque bande de fréquence Mel.
   - Enfin, on applique une Transformée en cosinus discrète inverse (IDCT) pour obtenir les coefficients cepstraux.

3. Étapes détaillées :
   - Transformée de Fourier : X(f) = ∫ x(t) * exp(-j * 2 * pi * f * t) dt
   - Spectre de puissance : |X(f)|^2
   - Banc de filtres Mel : Les fréquences sont converties en échelle Mel, qui est plus dense à basse fréquence et plus espacée à haute fréquence.
   - Logarithme de l'énergie : log(E_mel)
   - Transformée en cosinus discrète inverse : MFCC = IDCT(log(E_mel))

4. Caractéristiques :
   - Les MFCC capturent les caractéristiques essentielles du timbre, comme la "couleur" du son.
   - Ils dissocient la source (excitation glottique) du filtre (conduit vocal) dans les signaux de parole.
   - Les coefficients sont décorrélés, ce qui permet un stockage minimal d'information (moyenne et écart-type).

5. Avantages :
   - Représentation compacte du spectre sonore.
   - Inspirée de la perception humaine des sons.
   - Robustes au bruit de fond.

6. Applications :
   - Reconnaissance automatique de la parole.
   - Identification du locuteur.
   - Analyse des émotions dans la voix (affective computing).
   - Analyse musicale et reconnaissance de genres musicaux.

Exemple quantitatif et interprétation :
Pour un enregistrement de parole, les valeurs des 13 premiers coefficients MFCC pourraient être :

MFCC1 = 12.34, MFCC2 = 5.67, MFCC3 = -3.45, MFCC4 = 2.01, MFCC5 = -1.23,
MFCC6 = 0.78, MFCC7 = -0.56, MFCC8 = 0.34, MFCC9 = -0.21, MFCC10 = 0.12,
MFCC11 = -0.09, MFCC12 = 0.05, MFCC13 = 0.02

Interprétation :
1. MFCC1 (12.34) : Valeur élevée, indiquant une forte énergie globale du signal.
2. MFCC2 (5.67) : Valeur positive significative, suggérant une pente spectrale positive (plus d'énergie dans les hautes fréquences que dans les basses).
3. MFCC3 (-3.45) : Valeur négative, indiquant une courbure concave dans le spectre.
4. MFCC4 à MFCC13 : Valeurs de plus en plus proches de zéro, représentant des détails plus fins du spectre.

Ces valeurs pourraient correspondre à un son voisé, comme une voyelle, avec une énergie significative dans les moyennes et hautes fréquences. La décroissance rapide des coefficients d'ordre supérieur suggère un spectre relativement lisse, typique de la parole.

En comparant ces MFCC à ceux d'autres sons, on pourrait :
- Identifier le locuteur (les MFCC varient selon les caractéristiques vocales individuelles).
- Reconnaître le phonème prononcé (chaque son de parole a un profil MFCC distinct).
- Détecter l'émotion du locuteur (les états émotionnels influencent subtilement les MFCC).

Dans l'analyse musicale, des MFCC similaires pourraient indiquer un instrument à timbre brillant, comme une trompette ou un violon dans un registre aigu.
""")
    
    return descripteurs

def tracer_toile_araignee(valeurs, categories):
    angles = np.linspace(0, 2*np.pi, len(categories), endpoint=False)
    
    valeurs = np.concatenate((valeurs, [valeurs[0]]))
    angles = np.concatenate((angles, [angles[0]]))
    
    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
    ax.plot(angles, valeurs)
    ax.fill(angles, valeurs, alpha=0.25)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories)
    ax.set_ylim(0, 1)
    plt.title("Diagramme en toile d'araignée du Tristimulus")
    plt.show()

def ouvrir_fichier():
    fichier = filedialog.askopenfilename(filetypes=[("Fichiers WAV", "*.wav")])
    if fichier:
        entree_fichier.delete(0, tk.END)
        entree_fichier.insert(0, fichier)

def afficher_info(titre, texte):
    fenetre_info = tk.Toplevel()
    fenetre_info.title(titre)
    fenetre_info.geometry("600x400")
    
    texte_widget = tk.Text(fenetre_info, wrap=tk.WORD, padx=10, pady=10)
    texte_widget.insert(tk.END, texte)
    texte_widget.config(state=tk.DISABLED)
    texte_widget.pack(expand=True, fill=tk.BOTH)
    
    bouton_fermer = tk.Button(fenetre_info, text="Fermer", command=fenetre_info.destroy)
    bouton_fermer.pack(pady=10)

def analyser_frequence():
    fichier = entree_fichier.get()
    if fichier:
        try:
            data, framerate = lire_wav(fichier)
            tracer_wave(data, framerate)
            frequence_dominante = determiner_frequence(data, framerate)
            resultat.set(f"Fréquence dominante : {frequence_dominante:.2f} Hz")
        except Exception as e:
            resultat.set(f"Erreur : {str(e)}")

def analyser_tristimulus():
    fichier = entree_fichier.get()
    if fichier:
        try:
            y, sr = lire_wav(fichier)
            t1, t2, t3 = calculer_tristimulus(y, sr)
            categories = ['Fondamental (T1)', 'Harmoniques 2-4 (T2)', 'Harmoniques 5+ (T3)']
            valeurs = [t1, t2, t3]
            tracer_toile_araignee(valeurs, categories)
            resultat.set(f"Tristimulus : T1={t1:.2f}, T2={t2:.2f}, T3={t3:.2f}")
        except Exception as e:
            resultat.set(f"Erreur : {str(e)}")

def analyser_descripteurs():
    fichier = entree_fichier.get()
    if fichier:
        try:
            y, sr = lire_wav(fichier)
            descripteurs = calculer_descripteurs(y, sr)
            frame_resultats = tk.Frame(fenetre)
            frame_resultats.grid(row=5, column=0, columnspan=3, pady=10)
            
            for i, (k, (v, unit, description)) in enumerate(descripteurs.items()):
                tk.Label(frame_resultats, text=f"{k} :").grid(row=i, column=0, sticky="e")
                if isinstance(v, np.ndarray):
                    v_str = ", ".join([f"{val:.2f}" for val in v])
                    tk.Label(frame_resultats, text=f"{v_str} {unit}").grid(row=i, column=1, sticky="w")
                else:
                    tk.Label(frame_resultats, text=f"{v:.2f} {unit}").grid(row=i, column=1, sticky="w")
                
                info_button = tk.Button(frame_resultats, text="?", command=lambda k=k, d=description: afficher_info(k, d))
                info_button.grid(row=i, column=2)
            
            resultat.set("Analyse terminée. Cliquez sur '?' pour plus d'informations sur chaque descripteur.")
        except Exception as e:
            resultat.set(f"Erreur : {str(e)}")

# Création de l'interface graphique
fenetre = tk.Tk()
fenetre.title("Analyse Audio")

frame = tk.Frame(fenetre, padx=10, pady=10)
frame.grid(row=0, column=0)

tk.Label(frame, text="Fichier audio :").grid(row=0, column=0, sticky="e")
entree_fichier = tk.Entry(frame, width=50)
entree_fichier.grid(row=0, column=1, padx=5)
tk.Button(frame, text="Parcourir", command=ouvrir_fichier).grid(row=0, column=2)

tk.Button(frame, text="Analyser Fréquence", command=analyser_frequence).grid(row=1, column=0, columnspan=3, pady=10)
tk.Button(frame, text="Analyser Tristimulus", command=analyser_tristimulus).grid(row=2, column=0, columnspan=3)
tk.Button(frame, text="Analyser Descripteurs de Timbre", command=analyser_descripteurs).grid(row=3, column=0, columnspan=3, pady=10)

resultat = tk.StringVar()
tk.Label(frame, textvariable=resultat, wraplength=400, justify=tk.LEFT).grid(row=4, column=0, columnspan=3, pady=10)

fenetre.mainloop()