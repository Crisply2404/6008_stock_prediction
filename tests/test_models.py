"""Check model interface and persistence / 模型接口与保存恢复检查。"""
import io
import unittest
import torch

from src.models.patchtst import PatchTSTRegressor


class PatchTests(unittest.TestCase):
    def test_shape_gradient_and_restore(self):
        kwargs = dict(window=12, channels=2, patch_len=4, stride=4,
                      d_model=8, n_heads=2, n_layers=1, d_ff=16, dropout=0)
        model = PatchTSTRegressor(**kwargs)
        x = torch.randn(5, 12, 2)
        y = model(x)
        self.assertEqual(y.shape, (5,))
        y.square().mean().backward()
        self.assertTrue(torch.isfinite(model.projection.weight.grad).all())
        buffer = io.BytesIO()
        torch.save(model.state_dict(), buffer)
        buffer.seek(0)
        restored = PatchTSTRegressor(**kwargs)
        restored.load_state_dict(torch.load(buffer, weights_only=True))
        model.eval()
        restored.eval()
        torch.testing.assert_close(model(x), restored(x))

    def test_patch_tail_not_silently_discarded(self):
        with self.assertRaises(ValueError):
            PatchTSTRegressor(window=13, channels=2, patch_len=4, stride=4)

    def test_input_channel_count_checked(self):
        model = PatchTSTRegressor(window=12, channels=2, patch_len=4, stride=4)
        with self.assertRaises(ValueError):
            model(torch.randn(3, 12, 3))


if __name__ == "__main__":
    unittest.main()
