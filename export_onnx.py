import torch
import torch.onnx
import timm


model = timm.create_model('convnext_nano', pretrained=True, num_classes=2)
checkpoint = torch.load('timm_convnext_best.pt', map_location=torch.device('cpu'))
model.load_state_dict(checkpoint['state_dict']) 

model.eval()
dummy_input = torch.zeros((1,3, 50, 50))

torch.onnx.export(model, dummy_input, "weights/timm_convnext_best.onnx", verbose=True, export_params=True)
