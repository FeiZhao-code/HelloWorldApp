import torch
import torchvision
from torch.utils.mobile_optimizer import optimize_for_mobile
from model_v2 import MobileNetV2

# 1
model = MobileNetV2(num_classes=5)
model.load_state_dict(torch.load("MobileNetV2.pth"))
# model = torchvision.models.mobilenet_v3_small(pretrained=True)
model.eval()
example = torch.rand(1, 3, 224, 224)
traced_script_module = torch.jit.trace(model, example)
optimized_traced_model = optimize_for_mobile(traced_script_module)
optimized_traced_model._save_for_lite_interpreter("app/src/main/assets/model4.pt")
