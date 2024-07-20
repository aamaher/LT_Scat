from DE_Functions import *
import numpy as np
import matplotlib.pyplot as plt

# Main script
Pt = pt_src()
Pt.SetFluoro(0,0,-5)

Im = Pt.GetImage()

plt.figure()
plt.imshow(Im, cmap='gray')
plt.show()
plt.title('Forward Data')