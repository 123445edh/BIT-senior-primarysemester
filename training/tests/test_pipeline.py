import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import tempfile
import unittest
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from data import bytes_tensor
from models import SwinClassifier
from train import evaluate


class PipelineTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_swin_forward_backward_and_channel_reduction(self):
        baseline = SwinClassifier()
        slim = SwinClassifier(embed_dim=24, num_heads=[2, 4, 8], mlp_ratio=3)
        self.assertLess(sum(p.numel() for p in slim.parameters()), sum(p.numel() for p in baseline.parameters()))
        for model in [baseline, slim]:
            logits = model(torch.rand(2, 1, 32, 32))
            self.assertEqual(tuple(logits.shape), (2, 9))
            loss = torch.nn.functional.cross_entropy(logits, torch.tensor([0, 8]))
            loss.backward()
            self.assertTrue(torch.isfinite(model.patch_embed.weight.grad).all())

    def test_invalid_heads_rejected(self):
        with self.assertRaises(ValueError):
            SwinClassifier(embed_dim=25, num_heads=[2, 4, 8])

    def test_bytes_aggregation_rounding_and_unknown(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'sample.bytes'
            # 每段2字节；(0+1)/2按half-up取1，??不计入均值。
            path.write_bytes(b'00401000 '+b'00 01 '*512+b'?? FF '*512)
            array = bytes_tensor(path).numpy().reshape(-1)*255
            np.testing.assert_allclose(array[:512], 1)
            np.testing.assert_allclose(array[512:], 255)
            path.write_bytes(b'00401000 ?? ??')
            with self.assertRaises(ValueError):
                bytes_tensor(path)

    def test_evaluate_denominators(self):
        class Identity(torch.nn.Module):
            def forward(self, x):
                return x
        inputs = torch.eye(9)*10
        loader = DataLoader(TensorDataset(inputs, torch.arange(9)), batch_size=4)
        result = evaluate(Identity(), loader, torch.device('cpu'))
        self.assertEqual(result['samples'], 9)
        self.assertEqual(result['accuracy'], 1)
        self.assertEqual(result['macro_f1'], 1)
        self.assertGreater(result['seconds'], 0)


if __name__ == '__main__':
    unittest.main()
