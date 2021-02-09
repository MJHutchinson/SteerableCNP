import torch
import torch.nn as nn

from einops import rearrange

from steer_cnp.utils import MLP


class CNP(nn.Module):
    def __init__(
        self,
        x_dim,
        context_dim,
        embedding_dim,
        prediction_dim,
        encoder_hidden_dims,
        x_encoder_hidden_dims,
        decoder_hidden_dims,
        covariance_activation_function,
        min_cov,
        batch_norm=False,
    ):
        super().__init__()

        if x_encoder_hidden_dims is None:
            self.x_encoder = nn.Sequential()
        else:
            self.x_encoder = MLP(
                x_dim,
                x_encoder_hidden_dims,
                x_encoder_hidden_dims[-1],
                batch_norm=batch_norm,
            )
            x_dim = x_encoder_hidden_dims[-1]

        self.encoder = MLP(
            x_dim + context_dim,
            encoder_hidden_dims,
            embedding_dim,
            batch_norm=batch_norm,
        )

        self.decoder = MLP(
            x_dim + embedding_dim,
            decoder_hidden_dims,
            prediction_dim + prediction_dim,
            batch_norm=batch_norm,
        )
        self.prediction_dim = prediction_dim
        self.covariance_activation_function = covariance_activation_function
        self.min_cov = min_cov

        # scale = 1.5
        # for m in self.encoder.modules():
        #     if isinstance(m, nn.Linear):
        #         m.weight.data = m.weight * scale
        # scale = 5
        # offset = 5
        # for m in self.decoder.modules():
        #     if isinstance(m, nn.Linear):
        #         m.weight.data = m.weight * scale
        #         m.bias.data = m.bias + offset

        # Hack to increase output variance of decoder
        # self.decoder[-1].weight.data = self.decoder[-1].weight.data * 10
        # self.decoder[-1].bias.data = self.decoder[-1].bias

    def encode(self, X_context, Y_context):
        return self.encoder(
            torch.cat([self.x_encoder(X_context.float()), Y_context], dim=-1)
        ).mean(dim=-2, keepdim=True)

    def decode(self, embeddings, X_target):
        decoded = self.decoder(
            torch.cat(
                [
                    embeddings.expand(-1, X_target.shape[-2], -1),
                    self.x_encoder(X_target.float()),
                ],
                dim=-1,
            ),
        )
        means = decoded[..., : self.prediction_dim]
        covs = self.covariance_activation_function(decoded[..., self.prediction_dim :])
        covs = rearrange(
            covs,
            "b n (d1 d2) -> b n d1 d2",
            d1=self.prediction_dim,
            d2=self.prediction_dim,
        )
        covs = covs + torch.eye(self.prediction_dim).to(decoded.device) * self.min_cov

        return means, covs

    def forward(self, X_context, Y_context, X_target):
        return self.decode(self.encode(X_context, Y_context), X_target)