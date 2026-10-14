import unittest

from src.data import FEATURES, generate_dataset


class MobileMoneyDataTests(unittest.TestCase):
    def test_generation_is_reproducible_and_has_required_fields(self):
        first = generate_dataset(rows=300)
        second = generate_dataset(rows=300)
        self.assertEqual(list(first.columns), list(second.columns))
        self.assertEqual(first.to_csv(index=False), second.to_csv(index=False))
        self.assertTrue(set(FEATURES).issubset(first.columns))
        self.assertIn("is_fraud", first.columns)
        self.assertGreater(first["is_fraud"].sum(), 0)
        self.assertLess(first["is_fraud"].mean(), 0.5)


if __name__ == "__main__":
    unittest.main()