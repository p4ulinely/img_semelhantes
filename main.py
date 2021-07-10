import cv2
import numpy as np

board_img = cv2.imread('tabuleiro.png', cv2.IMREAD_GRAYSCALE)
knight_img = cv2.imread('cavalo_p.png', cv2.IMREAD_GRAYSCALE)

# knight_img = cv2.imread('cavalo_p2.png', cv2.IMREAD_GRAYSCALE)

# cv2.imshow('board_img', board_img)
# cv2.waitKey()
# cv2.destroyAllWindows()
# # cv2.imshow('knight_img', knight_img)
# cv2.waitKey()
# cv2.destroyAllWindows()

# TM_CCOEFF_NORMED
result = cv2.matchTemplate(board_img, knight_img, cv2.TM_CCOEFF_NORMED)

min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
w = knight_img.shape[1]
h = knight_img.shape[0]

# cv2.rectangle(board_img, max_loc, (max_loc[0]+w, max_loc[1]+h), (0,255,255), 2)

# cv2.imshow('board', board_img)
# cv2.waitKey()
# cv2.destroyAllWindows()

###########################
threshold = 0.95

yloc, xloc = np.where(result >= threshold)
print(len(xloc)) # quantidade de imagens semelhantes

for (x, y) in zip(xloc, yloc):
    cv2.rectangle(board_img, (x, y), (x+w, y+h), (0,255,255), 2)

cv2.imshow('board', board_img)
cv2.waitKey()
cv2.destroyAllWindows()