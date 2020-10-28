import torch
from model import Generator
import matplotlib.pyplot as plt
import numpy as np

# ----------------------------------------------------------------------------------------------
# モデル定義
generator = Generator()
model_path = 'trained_generator.pth'
generator.load_state_dict(torch.load(model_path,map_location=torch.device('cpu')))
generator.eval()


fig=plt.figure()
tate=5
yoko=6
for i in range(1,tate*yoko+1):
    ax=fig.add_subplot(tate,yoko,i)
    fake=generator(torch.rand((1,1,100)))
    fake=fake.detach().numpy().reshape(3,32,32).transpose(1,2,0)
    ax.tick_params(labelbottom=False,
                   labelleft=False,
                   labelright=False,
                   labeltop=False)
    ax.tick_params(bottom=False,
                   left=False,
                   right=False,
                   top=False)
    ax.imshow((fake+1)/2)

plt.savefig("generated_img.png")
plt.show()
