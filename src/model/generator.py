from torch import nn


class SoundStream(nn.Module):
    def __init__(self, embedding_dim, encoder_channels, decoder_channels, encoder, quantizer, decoder):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.encoder_channels = encoder_channels
        self.decoder_channels = decoder_channels
        self.encoder = encoder
        self.quantizer = quantizer
        self.decoder = decoder

    def forward(self, data_object, **batch):
        x = self.encoder(data_object)
        z_q, codes = self.quantizer(x)
        x_hat = self.decoder(z_q)
        return {"result": x_hat, "encoded": x, "quantized": z_q, "codes": codes}

    def __str__(self):
        """
        Model prints with the number of parameters.
        """
        all_parameters = sum([p.numel() for p in self.parameters()])
        trainable_parameters = sum([p.numel() for p in self.parameters() if p.requires_grad])

        result_info = super().__str__()
        result_info = result_info + f"\nAll parameters: {all_parameters}"
        result_info = result_info + f"\nTrainable parameters: {trainable_parameters}"

        return result_info