import pytest
import torch

from lewm.model import LeWM, count_params


@pytest.fixture(scope="module")
def model():
    torch.manual_seed(0)
    return LeWM()


def test_shapes(model):
    frames = torch.randint(0, 256, (2, 4, 3, 224, 224), dtype=torch.uint8)
    actions = torch.randn(2, 4, 10)
    emb, pred = model(frames, actions)
    assert emb.shape == (2, 4, 192) and pred.shape == (2, 3, 192)
    for T in (1, 2):
        assert model.predict(emb[:, :T], actions[:, :T]).shape == (2, T, 192)


def test_param_counts(model):
    enc, pred = count_params(model.encoder), count_params(model.predictor)
    print(f"\nencoder {enc / 1e6:.2f}M, predictor {pred / 1e6:.2f}M, "
          f"action encoder {count_params(model.action_encoder) / 1e6:.3f}M")
    assert 4.5e6 < enc < 6.5e6
    assert 9e6 < pred < 13e6


def test_predictor_causal(model):
    model.eval()
    # Non-zero adaLN weights, so actions actually reach the output (zero-init would hide a bug).
    with torch.no_grad():
        for blk in model.predictor.blocks:
            blk.ada.weight.normal_(std=0.2)
    torch.manual_seed(1)
    emb, act = torch.randn(3, 3, 192), torch.randn(3, 3, 10)
    emb2, act2 = emb.clone(), act.clone()
    # Random (not constant) perturbations: a constant shift is erased by LayerNorm.
    emb2[:, 2] += torch.randn(3, 192)
    act2[:, 2] += torch.randn(3, 10)
    with torch.no_grad():
        base = model.predict(emb, act)
        for e, a in ((emb2, act), (emb, act2)):
            out = model.predict(e, a)
            torch.testing.assert_close(out[:, :2], base[:, :2])
            # Position 2 must respond, well above float noise (~1e-7). A non-causal predictor
            # leaks ~6e-4 into positions 0-1, which assert_close above catches.
            assert (out[:, 2] - base[:, 2]).abs().max() > 1e-4
