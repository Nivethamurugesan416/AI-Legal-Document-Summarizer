import unittest
from src.summarization import chunk_text, word_count
from src.document_processing import extract_text

class SummarizerTests(unittest.TestCase):
    def test_chunk_overlap(self):
        self.assertEqual(chunk_text('a b c d e', 3, 1), ['a b c', 'c d e', 'e'])

    def test_empty_chunks(self):
        self.assertEqual(chunk_text(''), [])

    def test_word_count(self):
        self.assertEqual(word_count('one two three'), 3)

    def test_txt_extraction(self):
        self.assertEqual(extract_text('sample.txt', b'Hello world'), 'Hello world')

    def test_invalid_extension(self):
        with self.assertRaises(ValueError):
            extract_text('data.csv', b'a,b')

if __name__ == '__main__':
    unittest.main()
