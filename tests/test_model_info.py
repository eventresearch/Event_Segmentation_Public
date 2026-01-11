"""
Quick test script to verify model info printing functionality.
"""
import torch
import sys
sys.path.append('event_seg')

from models.smp_unet import SMPUNet, SMPDualEncoderUNet, CustomSMPUNet
from utils.utilities import print_model_info

# Test 1: Standard SMP UNet
print("\n### Test 1: Standard SMP UNet ###")
config1 = {
    'model_type': 'smp_unet',
    'encoder_name': 'resnet34',
    'encoder_weights': 'imagenet',
    'num_of_in_channels': 3,
    'num_of_out_classes': 11,
    'use_rgb': True,
    'use_event': False,
    'use_autoencoder': False
}
model1 = SMPUNet(config1)
print_model_info(model1, config1)

# Test 2: Custom SMP UNet
print("\n### Test 2: Custom SMP UNet ###")
config2 = {
    'model_type': 'custom_smp_unet',
    'encoder_name': 'resnet34',
    'encoder_weights': 'imagenet',
    'num_of_in_channels': 3,
    'num_of_out_classes': 11,
    'use_rgb': True,
    'use_event': False,
    'use_autoencoder': False
}
model2 = CustomSMPUNet(config2)
print_model_info(model2, config2)

# Test 3: Dual Encoder UNet with concat fusion
print("\n### Test 3: Dual Encoder UNet (concat fusion) ###")
config3 = {
    'model_type': 'smp_dual_encoder_unet',
    'encoder_name': 'resnet34',
    'encoder_weights': 'imagenet',
    'num_of_in_channels': 3,
    'event_channels': 1,
    'num_of_out_classes': 11,
    'fusion_type': 'concat',
    'event_ms': 50,
    'use_rgb': True,
    'use_event': True,
    'use_autoencoder': False
}
model3 = SMPDualEncoderUNet(config3)
print_model_info(model3, config3)

# Test 4: Dual Encoder UNet with add fusion
print("\n### Test 4: Dual Encoder UNet (add fusion) ###")
config4 = {
    'model_type': 'smp_dual_encoder_unet',
    'encoder_name': 'efficientnet-b0',
    'encoder_weights': 'imagenet',
    'num_of_in_channels': 3,
    'event_channels': 1,
    'num_of_out_classes': 11,
    'fusion_type': 'add',
    'event_ms': 100,
    'use_rgb': True,
    'use_event': True,
    'use_autoencoder': False
}
model4 = SMPDualEncoderUNet(config4)
print_model_info(model4, config4)

print("\n### All tests completed successfully! ###\n")
