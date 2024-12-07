import torchreid

# Download and specify model to download
model_name = 'osnet_x1_0'
model = torchreid.models.build_model(
    name=model_name, num_classes=1000, loss='softmax'
)

# Download the pre-trained weights for the model
torchreid.utils.load_pretrained_weights(model, model_name)
