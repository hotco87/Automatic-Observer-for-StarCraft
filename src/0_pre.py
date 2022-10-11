import sys
import torch
print(sys.version)
print(torch.__file__)
print(torch.__version__)
print(torch.cuda.is_available())
print(torch.cuda.current_device())

# with open('/opt/conda/lib/python3.8/site-packages/torchvision/models/detection/transform.py', 'r') as file :
#   filedata = file.read()
#
# filedata = filedata.replace('image = self.normalize(image)', '')
#
# with open('/opt/conda/lib/python3.8/site-packages/torchvision/models/detection/transform.py', 'w') as file:
#   file.write(filedata)
