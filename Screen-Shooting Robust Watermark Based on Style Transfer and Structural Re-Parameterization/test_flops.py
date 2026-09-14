import pytest
import torch
from calflops import calculate_flops
from torchvision.models import resnet50
import model
import Noise_Layer
# from thop import profile
# from thop import clever_format


def test_flops_calculation():
    model1 = model.Discriminator(64)
    # model1 = model.U_Net_Encoder_Diffusion()
    # model1 = model.Decoder()
    # model1 = Noise_Layer.ScreenShooting()

    input1 = (1, 3, 128, 128)
    flops, macs, params = calculate_flops(model1, input_shape=input1)
    print(flops, macs, params)



if __name__ == "__main__":
    pytest.main()