import unittest

from embedforge.utils.yamlish import parse


class TestYamlish(unittest.TestCase):
    def test_scalars(self):
        d = parse("a: 1\nb: 2.5\nc: true\nd: hello\ne: 'x'\n")
        self.assertEqual(d["a"], 1)
        self.assertEqual(d["b"], 2.5)
        self.assertIs(d["c"], True)
        self.assertEqual(d["d"], "hello")
        self.assertEqual(d["e"], "x")

    def test_nested_and_list(self):
        d = parse("model:\n  dim: 64\n  mrl_dims: [8, 16, 32]\n")
        self.assertEqual(d["model"]["dim"], 64)

    def test_block_list(self):
        d = parse("keywords:\n  - a\n  - b\n")
        self.assertEqual(d["keywords"], ["a", "b"])

    def test_inline_comment(self):
        d = parse("name: mock  # comment\n")
        self.assertEqual(d["name"], "mock")


if __name__ == "__main__":
    unittest.main()
