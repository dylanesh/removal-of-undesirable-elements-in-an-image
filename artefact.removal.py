import imageio
import Create_frame as cf
import numpy as np
import scipy.fftpack as ft
import scipy.ndimage as nd
from PIL import Image
import matplotlib.pyplot as plt
adata = imageio.imread( 'Example_baseball.tiff')
print(adata)
#zdata = 0.5 *adata[:,:,0] + 0.75*adata[:,:,1] + 0.25*adata[:,:,2]   Grayscale for .jgp
print("Dimensiones de la imagen:", adata.shape)
V,H = adata.shape     #Assign dimensions
frame2 = cf.Plop(adata, (2*V,2*H))   #   larger frame
fouriert=ft.fft2 (frame2/adata.max())   #    fourier transform and decentralized frecuencies
cft = ft.fftshift(fouriert)   #   centralized frecuencies
mask = np.ones( (2*V,2*H))
mask[V-8:V+8,:666] = 0
mask[V-8:V+8,-666:] = 0    #   Masking(15:19)
mask[:450,H-8:H+8] = 0
mask[-450:,H-8:H+8] = 0
mask_Gau = nd.gaussian_filter(mask, 10)   #   applying gaussian filter to mask
dft=ft.fftshift(mask_Gau * cft)    #   decentralized frecuencies
ifouriert = ft.ifft2(dft)    #   inverse fourier transform
bdata = ifouriert.real[V-242:V+242,H-363:H+363]   #   real size

#                                PLOTS
# 1. original image
plt.imshow(adata, "gray"), plt.title("Original")
plt.show()
# 2. spectrum
plt.imshow(np.log(1+np.abs(fouriert)), "gray"), plt.title("Espectro")
plt.show()
# 3. centralized spectrum
plt.imshow(np.log(1+np.abs(cft)), "gray"), plt.title("Centralizado")
plt.show()
# 4. mask wt gaussian filter
plt.imshow(mask_Gau, "gray"), plt.title("Mascara")
plt.show()
# 5. spectrum wt filter applied
plt.imshow(np.log(1+np.abs(dft)), "gray"), plt.title("Filtrado")
plt.show()
#6. processed image
plt.imshow(bdata, "gray"), plt.title("Imagen procesada")
plt.show()
