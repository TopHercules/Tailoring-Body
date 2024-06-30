import cv2
import numpy as np
import torch
from torchvision import transforms
from network import UNet as HUNet

def predict(filepath):
    RES = 128
    model_h = HUNet(128)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    pretrained_model_h = torch.load('model/best.pth', map_location=device)
    if torch.cuda.is_available():
        model_h.load_state_dict(pretrained_model_h["state_dict"])

    if torch.cuda.is_available():
        model = model_h.cuda(0)
    else:
        model = model_h

    assert ".jpg" in filepath or ".png" in filepath or ".jpeg" in filepath, "Please use .jpg or .png format"
    X = cv2.cvtColor(cv2.imread(filepath), cv2.COLOR_BGR2RGB).astype('float32')
    scale = RES / max(X.shape[:2])

    X_scaled = cv2.resize(X, (0,0), fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)

    if X_scaled.shape[1] > X_scaled.shape[0]:
        p_a = (RES - X_scaled.shape[0])//2
        p_b = (RES - X_scaled.shape[0])-p_a
        X = np.pad(X_scaled, [(p_a, p_b), (0, 0), (0,0)], mode='constant')

    elif X_scaled.shape[1] <= X_scaled.shape[0]:
        p_a = (RES - X_scaled.shape[1])//2
        p_b = (RES - X_scaled.shape[1])-p_a
        X = np.pad(X_scaled, [(0, 0), (p_a, p_b), (0,0)], mode='constant')

    o_img = X.copy()
    X /= 255
    X = transforms.ToTensor()(X).unsqueeze(0)

    if torch.cuda.is_available():
        X = X.cuda()
        
    model.eval()
    with torch.no_grad():
        _, _, h_p = model(X)

    return h_p.item() * 100