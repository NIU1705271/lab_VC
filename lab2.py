import cv2
import numpy as np
import time
import sys
import os
from scipy.signal import correlate2d


def align_image_phase_correlation(img1, img2):
    """
    Alinea la segona imatge amb la primera utilitant correlació de fase
    Retorna la imatge alineada i el desplaçament calculat
    """
    # Crea la finestra de Hann per reduir l'efecte de les vores de les imatges
    h, w = img1.shape
    hann_y = np.hanning(h)
    hann_x = np.hanning(w)
    window = np.outer(hann_y, hann_x)

    # Apliquem la finestra
    img1_w = img1 * window
    img2_w = img2 * window

    # Calculem el la transformació de fourier (FFT) 2D
    F_img1 = np.fft.fft2(img1_w)
    F_img2 = np.fft.fft2(img2_w)
    
    # Calculem el espectre cross_power
    cross_power_spectrum = (F_img1 * np.conj(F_img2)) / np.abs(F_img1 * np.conj(F_img2) + 1e-10)

    # Calculem la inversa del FFT per obtenir el mapa de correlació de fase
    r = np.fft.ifft2(cross_power_spectrum)
    r = np.abs(r)

    # Trobem el pic de la correlació
    shift_y, shift_x = np.unravel_index(np.argmax(r), r.shape)

    # Gestionem que el desplaçament no sigui més gran que el limit
    if shift_y > h // 2:
        shift_y -= h
    if shift_x > w // 2:
        shift_x -= w
        
    # Apliquem el deplaçament
    rows, cols = img2.shape
    M = np.float32([[1, 0, shift_x], [0, 1, shift_y]])
    aligned_img2 = cv2.warpAffine(img2, M, (cols, rows))
    
    return aligned_img2, (shift_x, shift_y)

"""
# Alternative Correlation Methods (Commented out as requested)

def align_image_spatial_correlation(img1, img2):
    """
    # 1. Correlació espacial pura lliscant una imatge sobre l'altra.
    # Molt lent computacionalment per a imatges grans.
    """
    h, w = img1.shape
    
    # Restem la mitjana per centrar els valors i evitar que la brillantor domini
    img1_norm = img1 - np.mean(img1)
    img2_norm = img2 - np.mean(img2)
    
    # Calculem la correlació espacial (pot trigar minuts depenent de la mida)
    print("   (Calculant correlació espacial... això pot trigar)")
    corr = correlate2d(img1_norm, img2_norm, mode='same')
    
    # Trobem el pic de màxima coincidència
    shift_y, shift_x = np.unravel_index(np.argmax(corr), corr.shape)
    
    # Ajustem el desplaçament respecte al centre de la imatge
    shift_y -= h // 2
    shift_x -= w // 2
    
    # Apliquem el desplaçament
    M = np.float32([[1, 0, shift_x], [0, 1, shift_y]])
    aligned_img2 = cv2.warpAffine(img2, M, (w, h))
    
    return aligned_img2, (shift_x, shift_y)

def align_image_fourier_correlation(img1, img2):
    """
    # 2. Correlació estàndard resolta mitjançant la Transformada de Fourier.
    # Més ràpid, però el pic pot ser difús.
    """
    h, w = img1.shape
    
    # Calculem la FFT 2D
    F_img1 = np.fft.fft2(img1)
    F_img2 = np.fft.fft2(img2)
    
    # Producte en l'espai de Fourier (SENSE normalitzar per la magnitud)
    cross_correlation = np.fft.ifft2(F_img1 * np.conj(F_img2))
    cross_correlation = np.abs(cross_correlation)
    
    # Trobem el pic
    shift_y, shift_x = np.unravel_index(np.argmax(cross_correlation), cross_correlation.shape)
    
    # Ajustem per l'efecte circular (wrap-around) de la FFT
    if shift_y > h // 2:
        shift_y -= h
    if shift_x > w // 2:
        shift_x -= w
        
    # Apliquem el desplaçament
    M = np.float32([[1, 0, shift_x], [0, 1, shift_y]])
    aligned_img2 = cv2.warpAffine(img2, M, (w, h))
    
    return aligned_img2, (shift_x, shift_y)
    
    def align_image_ncc(img1, img2):
    """
    # 4. Correlació Creuada Normalitzada (NCC) utilitzant OpenCV.
    # Afegeix padding a la imatge base per permetre trobar desplaçaments.
    """
    h, w = img1.shape
    
    # Afegim un marge (padding) a img1. Assumim que el desplaçament màxim 
    # no serà superior al 20% de la mida de la imatge.
    pad_y, pad_x = h // 5, w // 5 
    img1_padded = cv2.copyMakeBorder(img1, pad_y, pad_y, pad_x, pad_x, cv2.BORDER_CONSTANT, value=0)
    
    # Busquem img2 dins de img1_padded utilitzant NCC
    res = cv2.matchTemplate(img1_padded, img2, cv2.TM_CCOEFF_NORMED)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
    
    # max_loc ens dona la cantonada superior esquerra del millor encaix.
    # Com que img1 estava desplaçada artificialment pel padding, ho compensem.
    shift_x = max_loc[0] - pad_x
    shift_y = max_loc[1] - pad_y
    
    M = np.float32([[1, 0, shift_x], [0, 1, shift_y]])
    aligned_img2 = cv2.warpAffine(img2, M, (w, h))
    
    return aligned_img2, (shift_x, shift_y)
"""

def autocrop_borders(img, crop_margin=0.10):
    """
    Elimina les vores detectant els costats de les imatges
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape

    # Detecció de costats Canny
    edges = cv2.Canny(gray, 50, 150)
    
    hw = int(w * crop_margin)
    hh = int(h * crop_margin)

    # Analitzem linies horitzontals per la part de baix i la de dalt de l'iamtge
    row_edge_sums = np.sum(edges, axis=1)
    top_cut = np.argmax(row_edge_sums[:hh])
    bottom_cut = h - hh + np.argmax(row_edge_sums[-hh:])
    
    # Analitzem les linies verticals per la part esquerra i dreta de les imatges
    col_edge_sums = np.sum(edges, axis=0)
    left_cut = np.argmax(col_edge_sums[:hw])
    right_cut = w - hw + np.argmax(col_edge_sums[-hw:])

    # Apliquem un marge de seguretat per amagar desperfectes de les bandes de color degut a l'alineament
    margin_y = int(h * 0.015)
    margin_x = int(w * 0.015)
    
    top = min(top_cut + margin_y, h // 3)
    bottom = max(bottom_cut - margin_y, 2 * h // 3)
    left = min(left_cut + margin_x, w // 3)
    right = max(right_cut - margin_x, 2 * w // 3)
    
    return img[top:bottom, left:right]

def process_image(filename):
    print(f"Processing {filename}...")
    
    # 1. Llegim la imatge en esacla de grisos
    img_path = os.path.join(r"c:\Users\llucf\Downloads\img_lab2", filename)
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    
    if img is None:
        print(f"Error: Could not read image {filename}")
        return
        
    # Comencem un temporitzador
    start_time = time.time()

    # 2. Dividim l'imatge en tres canals
    h, w = img.shape
    h_channel = h // 3
    
    b = img[0:h_channel, :]
    g = img[h_channel:2*h_channel, :]
    r = img[2*h_channel:3*h_channel, :]
    
    # Eliminem les vores (15%) per un millor alineament
    crop_h = int(h_channel * 0.15)
    crop_w = int(w * 0.15)
    
    b_cropped = b[crop_h:-crop_h, crop_w:-crop_w]
    g_cropped = g[crop_h:-crop_h, crop_w:-crop_w]
    r_cropped = r[crop_h:-crop_h, crop_w:-crop_w]

    # 3. Alineem G,R i B
    print("Aligning Green channel...")
    g_aligned_cropped, shift_g = align_image_phase_correlation(b_cropped, g_cropped)
    
    print("Aligning Red channel...")
    r_aligned_cropped, shift_r = align_image_phase_correlation(b_cropped, r_cropped)
    
    print(f"Shift G (x,y): {shift_g}")
    print(f"Shift R (x,y): {shift_r}")

    # Apliquem els desplaçaments als canal originals (sense retall)
    M_g = np.float32([[1, 0, shift_g[0]], [0, 1, shift_g[1]]])
    g_aligned = cv2.warpAffine(g, M_g, (w, h_channel))
    
    M_r = np.float32([[1, 0, shift_r[0]], [0, 1, shift_r[1]]])
    r_aligned = cv2.warpAffine(r, M_r, (w, h_channel))
    
    # Juntem els canals a una sola imatge RGB
    color_img = cv2.merge([b, g_aligned, r_aligned])

    # Retallem les vores
    print("Cropping borders...")
    color_img = autocrop_borders(color_img)

    # Acabem el temporitzador
    end_time = time.time()
    print(f"Processing time: {end_time - start_time:.4f} seconds")

    # 4. Guardem el resultat
    base_name, ext = os.path.splitext(filename)
    output_filename = f"{base_name}_color{ext}"
    output_path = os.path.join(r"c:\Users\llucf\Downloads\img_lab2", output_filename)
    
    cv2.imwrite(output_path, color_img)
    print(f"Saved aligned image to {output_filename}\n")
    

    display_img = cv2.resize(color_img, (0,0), fx=0.3, fy=0.3)
    cv2.imshow(f"Aligned {filename}", display_img)


if __name__ == "__main__":
    folder = r"c:\Users\llucf\Downloads\img_lab2"
    # Trobem els fitxers .jpg que no son imatges que ja estiguin a color
    files = [f for f in os.listdir(folder) if f.endswith('.jpg') and not f.endswith('_color.jpg')]
    files.sort()
    
    for f in files:
        process_image(f)
