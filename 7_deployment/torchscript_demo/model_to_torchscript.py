import torch
import torch.nn.functional as F

import torchvision.models as models

r18 = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

r18_scripted = torch.jit.script(r18)
print(r18_scripted.graph)
dummy_input = torch.rand(1, 3, 224, 224)

r18_traced = torch.jit.trace(r18, dummy_input)
print(r18_traced.graph)
print(r18_traced.code)
unscripted_output = r18(dummy_input)
scripted_output = r18_scripted(dummy_input)

unscripted_top5 = F.softmax(unscripted_output, dim=1).topk(5).indices
scripted_top5 = F.softmax(scripted_output, dim=1).topk(5).indices
print(f'Python model top 5 results:\n {unscripted_top5}')
print(f'TorchScript model top 5 results:\n {scripted_top5}')

r18_scripted.save('r18_scripted.pt')
print(type(r18_scripted))
