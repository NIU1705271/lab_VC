# -*- coding: utf-8 -*-
"""
Created on Wed Mar  4 18:15:06 2026

@author: llucf
"""

import os
import sys
print("=== PYTHON EXECUTABLE IN USE ===")
print(sys.executable)
print("================================")
import glob
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Funcions auxiliars per emular MATLAB
def strel_disk(radius):
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*radius+1, 2*radius+1))

# =========================================================================
# TASCA 1 i 2 - Mitjana i STD
# =========================================================================
print('TASCA 1 i 2')
folderPath = os.path.join('lab1', 'img', 'train')
files = sorted(glob.glob(os.path.join(folderPath, '*.jpg')))

all_images_grey = []

# Llegim les imatges de train
for f in files:
    img = cv2.imread(f)
    img_grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    all_images_grey.append(img_grey)

# Calculem la matriu, mitjana i std
matriu = np.stack(all_images_grey, axis=2).astype(float)
imatge_mitjana_double = np.mean(matriu, axis=2)
imatge_std_double = np.std(matriu, axis=2, ddof=1)

imatge_mitjana = imatge_mitjana_double.astype(np.uint8)
imatge_std = imatge_std_double.astype(np.uint8)

# Mostrem el resultat
plt.figure(num='Tasca 2: Model de Fons')
plt.subplot(1, 2, 1); plt.imshow(imatge_mitjana, cmap='gray'); plt.title('Mitjana')
plt.subplot(1, 2, 2); plt.imshow(imatge_std, cmap='gray'); plt.title('Std')
plt.show(block=False)

# =========================================================================
# TASCA 3 - Detecció utilitzant l'std com a referència
# =========================================================================
print('\nTASCA 3')
sumatori = np.sum(imatge_std)
mida = imatge_std.size
mitjana_total = sumatori / mida
llindar = 1.1 * mitjana_total
imatge_std_senseFons = (imatge_std > llindar).astype(np.uint8)

num_imatges_t3 = 10
total_imatges = len(files)
# indexs aleatoris
indexs_aleatoris_t3 = np.random.permutation(total_imatges)[:num_imatges_t3]

imatges_mostra_t3 = []
imatges_finals_t3 = []

for i, idx_triat in enumerate(indexs_aleatoris_t3):
    img_original = all_images_grey[idx_triat]
    imatges_mostra_t3.append(img_original)
    
    img_final = img_original * imatge_std_senseFons
    imatges_finals_t3.append(img_final)
    
    plt.figure(num=f'Tasca 3: Mostra {i+1}')
    plt.subplot(1, 2, 1)
    plt.imshow(img_original, cmap='gray')
    plt.title('Imatge Original (Grisos)')
    
    plt.subplot(1, 2, 2)
    plt.imshow(img_final, cmap='gray')
    plt.title('Cotxe Aïllat (Fons Negre)')
plt.show(block=False)

# =========================================================================
# TASCA 4 - Subtracció del fons usant la Mitjana, Std, Alpha i Beta
# =========================================================================
print('\nTASCA 4')
num_imatges_t4 = 5
indexs_aleatoris_t4 = np.random.permutation(total_imatges)[:num_imatges_t4]
alpha_t4 = 1.0
beta_t4 = 8.0

plt.figure(num=f'Tasca 4: Resultats per Alpha = {alpha_t4:.1f}, Beta = {beta_t4:.1f}', figsize=(8, 12))

for i, idx_triat in enumerate(indexs_aleatoris_t4):
    img_original = all_images_grey[idx_triat]
    
    # Calculem la diferència
    img_gray_double = img_original.astype(float)
    diferencia_act = np.abs(img_gray_double - imatge_mitjana_double)
    
    # Apliquem el llindar i els filtres
    mask_elaborat_act = diferencia_act > (alpha_t4 * imatge_std_double + beta_t4)
    
    # Dibuixem els subplots en una graella
    plt.subplot(5, 2, 2*i + 1)
    plt.imshow(img_original, cmap='gray')
    if i == 0:
        plt.title('Imatge Original')
        
    plt.subplot(5, 2, 2*i + 2)
    plt.imshow(mask_elaborat_act, cmap='gray')
    if i == 0:
        plt.title(f'Aïllat ($\\alpha={alpha_t4:.1f}$, $\\beta={beta_t4:.1f}$)')

plt.tight_layout()
plt.show(block=False)

# =========================================================================
# TASCA 5 - Gravació del Vídeo amb Morfologia
# =========================================================================
print('\nTASCA 5')
millor_alpha_t5 = 1.0
millor_beta_t5 = 8.0
se_neteja_t5 = strel_disk(2)
se_omplir_t5 = strel_disk(15)

video_filename = 'resultat_cotxes.mp4'
h, w = all_images_grey[0].shape
fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
v = cv2.VideoWriter(video_filename, fourcc, 15.0, (w, h), isColor=False)

for k, img_grey in enumerate(all_images_grey):
    img_gray_double = img_grey.astype(float)
    
    diferencia_act = np.abs(img_gray_double - imatge_mitjana_double)
    mask_bruta = (diferencia_act > (millor_alpha_t5 * imatge_std_double + millor_beta_t5)).astype(np.uint8)
    
    # Morfologia
    mask_neta_video = cv2.morphologyEx(mask_bruta, cv2.MORPH_OPEN, se_neteja_t5)
    mask_neta_video = cv2.morphologyEx(mask_neta_video, cv2.MORPH_CLOSE, se_omplir_t5)
    
    frame_video = mask_neta_video * 255 
    v.write(frame_video)

v.release()
print(f'Vídeo guardat correctament: {video_filename}')

# =========================================================================
# TASCA 6 - Avaluació i Grid Search 4D
# =========================================================================
print('\nTASCA 6')
folder_test = os.path.join('lab1', 'img', 'test')
folder_gt = os.path.join('lab1', 'img', 'groundtruth')

files_test = sorted(glob.glob(os.path.join(folder_test, '*.jpg')))
files_gt = sorted(glob.glob(os.path.join(folder_gt, '*.png')))
num_imatges_test = len(files_test)

best_alpha = 1.0
best_beta = 8.0
best_rn = 2
best_ro = 15

print('\nParàmetres fixats aplicats:')
print(f'Alpha  : {best_alpha:.1f}')
print(f'Beta   : {best_beta:.1f}')
print(f'Neteja : {best_rn}')
print(f'Omplir : {best_ro}\n')

acc_cas1 = np.zeros(num_imatges_test)
acc_cas2 = np.zeros(num_imatges_test)
acc_cas3 = np.zeros(num_imatges_test)

se_neteja_optim = strel_disk(best_rn)
se_omplir_optim = strel_disk(best_ro)
se_exagerat = strel_disk(best_ro + 10)

print('Calculant les mètriques per als 3 casos...')
for i in range(num_imatges_test):
    # Llegim les imatges
    img_test_bgr = cv2.imread(files_test[i])
    img_test = cv2.cvtColor(img_test_bgr, cv2.COLOR_BGR2GRAY).astype(float)
    
    gt_img = cv2.imread(files_gt[i], cv2.IMREAD_GRAYSCALE)
    mask_gt = (gt_img == 255)
    
    diferencia = np.abs(img_test - imatge_mitjana_double)
    mask_base_optima = diferencia > (best_alpha * imatge_std_double + best_beta)
    
    # CAS 1: Només els paràmetres base (sense filtres)
    acc_cas1[i] = np.sum(mask_base_optima == mask_gt) / mask_gt.size
    
    # Convertim a uint8 per a l'OpenCV
    mask_base_optima_uint8 = mask_base_optima.astype(np.uint8)
    
    # CAS 2: Resultats amb els filtres (Neteja=2, Omplir=15)
    mask_cas2 = cv2.morphologyEx(mask_base_optima_uint8, cv2.MORPH_OPEN, se_neteja_optim)
    mask_cas2 = cv2.morphologyEx(mask_cas2, cv2.MORPH_CLOSE, se_omplir_optim)
    acc_cas2[i] = np.sum(mask_cas2.astype(bool) == mask_gt) / mask_gt.size
    
    # CAS 3: Aplicant un filtre exagerat per veure com empitjora (radi 25)
    mask_cas3 = cv2.morphologyEx(mask_base_optima_uint8, cv2.MORPH_OPEN, se_exagerat)
    mask_cas3 = cv2.morphologyEx(mask_cas3, cv2.MORPH_CLOSE, se_exagerat)
    acc_cas3[i] = np.sum(mask_cas3.astype(bool) == mask_gt) / mask_gt.size
    
    if (i + 1) % 20 == 0:
        print(f'  Processades {i + 1} de {num_imatges_test} imatges test...')

# --- 3. MOSTRAR RESULTATS DELS 3 CASOS ---
print('\nRESULTATS DELS 3 CASOS CONCRETS')
print(f'Cas 1 (Sense Filtres: a={best_alpha:.1f}, b={best_beta:.1f})                     : {np.mean(acc_cas1) * 100:.2f} %')
print(f'Cas 2 (Amb filtres: Neteja={best_rn}, Omplir={best_ro})       : {np.mean(acc_cas2) * 100:.2f} %')
print(f'Cas 3 (Filtres Exagerats: Radi {best_ro + 10})  : {np.mean(acc_cas3) * 100:.2f} %')
print('---------------------------------------------')

plt.show()
