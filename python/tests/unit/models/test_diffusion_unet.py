import torch

from python.src.models.probabilistic.diffusion_unet import (
    DiffusionUNet1D,
    ResidualBlock1D,
    SinusoidalPositionEmbeddings,
)


def test_sinusoidal_position_embeddings():
    dim = 64
    batch_size = 4
    embedder = SinusoidalPositionEmbeddings(dim)
    t = torch.linspace(0, 100, batch_size)
    embeddings = embedder(t)

    assert embeddings.shape == (batch_size, dim)
    # Check that different times produce different embeddings
    assert not torch.allclose(embeddings[0], embeddings[1])


def test_residual_block_1d_no_time():
    in_c, out_c = 16, 32
    block = ResidualBlock1D(in_c, out_c)
    x = torch.randn(2, in_c, 10)
    out = block(x)
    assert out.shape == (2, out_c, 10)


def test_residual_block_1d_with_time():
    in_c, out_c, t_dim = 16, 32, 64
    block = ResidualBlock1D(in_c, out_c, time_emb_dim=t_dim)
    x = torch.randn(2, in_c, 10)
    t_emb = torch.randn(2, t_dim)
    out = block(x, time_emb=t_emb)
    assert out.shape == (2, out_c, 10)


def test_diffusion_unet_1d_forward():
    batch, seq, feat = 2, 32, 4
    model = DiffusionUNet1D(
        input_dim=feat, output_dim=feat, hidden_dim=32, layers=[1, 2]
    )

    x = torch.randn(batch, seq, feat)
    t = torch.randint(0, 100, (batch,))

    out = model(x, t)
    assert out.shape == (batch, seq, feat)


def test_diffusion_unet_1d_with_condition():
    batch, seq, feat_x, feat_cond = 2, 32, 4, 2
    # Input dim should be feat_x + feat_cond because they are concatenated
    model = DiffusionUNet1D(
        input_dim=feat_x + feat_cond, output_dim=feat_x, hidden_dim=32
    )

    x = torch.randn(batch, seq, feat_x)
    cond = torch.randn(batch, seq, feat_cond)
    t = torch.randint(0, 100, (batch,))

    out = model(x, t, cond=cond)
    assert out.shape == (batch, seq, feat_x)


def test_diffusion_unet_1d_uneven_seq_len():
    # Test if interpolation handles seq_len that doesn't divide perfectly by 2 (e.g. 33)
    batch, seq, feat = 2, 33, 4
    model = DiffusionUNet1D(
        input_dim=feat, output_dim=feat, hidden_dim=32, layers=[1, 2]
    )

    x = torch.randn(batch, seq, feat)
    t = torch.randint(0, 100, (batch,))

    out = model(x, t)
    assert out.shape == (batch, seq, feat)
