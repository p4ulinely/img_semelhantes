import cv2
import numpy as np

tabuleiro = cv2.imread('tabuleiro.png', cv2.IMREAD_UNCHANGED)
cavalo_preto = cv2.imread('cavalo_p.png', cv2.IMREAD_UNCHANGED)

# cv2.imshow('meu tabuleiro', tabuleiro)
# cv2.imshow('meu tabuleiro', cavalo_preto)
# cv2.waitKey()

local_no_tabuleiro_parecido_com_o_cavalo = cv2.matchTemplate(tabuleiro, cavalo_preto, cv2.TM_CCOEFF_NORMED)

min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(local_no_tabuleiro_parecido_com_o_cavalo)

largura = cavalo_preto.shape[1]
altura = cavalo_preto.shape[0]

# cv2.rectangle(tabuleiro, max_loc, (max_loc[0]+largura, max_loc[1]+altura), (0,0,255), 2)
# cv2.imshow('meu tabuleiro - com cavalos marcados', tabuleiro)
# cv2.waitKey()

limiar = 0.93
yloc, xloc = np.where(local_no_tabuleiro_parecido_com_o_cavalo >= limiar)
# print(len(xloc)) # achou apenas 2 lugares semelhantes

for (x,y) in zip(xloc, yloc):
    cv2.rectangle(tabuleiro, (x,y), (x+largura, y+altura), (0,0,255), 2)

cv2.imshow('meu tabuleiro - com cavalos pretos marcados', tabuleiro)
cv2.waitKey()
cv2.destroyAllWindows()