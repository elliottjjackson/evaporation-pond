import unittest
import main  # Importing your main script


class TestMainScript(unittest.TestCase):
    def setUp(self):
        # Setup code if necessary, e.g., initializing main.py with input file
        main.run_script("input.txt", "output/output.txt")  # Example function call

    def test_output(self):
        with open("test_output.txt", "r") as expected, open(
            "output/output.txt", "r"
        ) as actual:
            self.assertEqual(
                expected.read(),
                actual.read(),
                "The actual output does not match the expected output.",
            )


if __name__ == "__main__":
    unittest.main()
