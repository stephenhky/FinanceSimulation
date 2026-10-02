import os
import tempfile
import unittest
from unittest.mock import patch, MagicMock

from click.testing import CliRunner
import pandas as pd

from finsim.cli import retrieve_stock_symbols


class TestRetrieveStockSymbolsCLI(unittest.TestCase):
    def setUp(self):
        self.runner = CliRunner()
        self.sample_symbols = [
            {'symbol': 'AAPL', 'mic': 'XNAS', 'type': 'Common Stock'},
            {'symbol': 'MSFT', 'mic': 'XNAS', 'type': 'Common Stock'},
            {'symbol': 'BRK.A', 'mic': 'XNYS', 'type': 'Common Stock'},
            {'symbol': 'TEST.PUB', 'mic': 'XNAS', 'type': 'PUBLIC'},
            {'symbol': 'OTHER', 'mic': 'BATS', 'type': 'Common Stock'},
        ]

    def test_help(self):
        result = self.runner.invoke(retrieve_stock_symbols, ['--help'])
        self.assertEqual(result.exit_code, 0)
        self.assertIn('Retrieve stock symbols from Finnhub', result.output)
        self.assertIn('--finnhubtokenpath', result.output)
        self.assertIn('--useenvtoken', result.output)
        self.assertIn('--shorten', result.output)

    def test_missing_outputpath(self):
        result = self.runner.invoke(retrieve_stock_symbols, [])
        self.assertNotEqual(result.exit_code, 0)
        self.assertIn('Missing argument', result.output)

    def test_missing_token_options(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            out_file = os.path.join(tmpdir, 'symbols.json')
            result = self.runner.invoke(retrieve_stock_symbols, [out_file])
            self.assertNotEqual(result.exit_code, 0)
            self.assertIsInstance(result.exception, ValueError)
            self.assertIn('Either --finnhubtokenpath or --useenvtoken must be specified', str(result.exception))

    def test_useenvtoken_missing_env(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            out_file = os.path.join(tmpdir, 'symbols.json')
            env = os.environ.copy()
            env.pop('FINNHUBTOKEN', None)
            result = self.runner.invoke(retrieve_stock_symbols, [out_file, '--useenvtoken'], env=env)
            self.assertNotEqual(result.exit_code, 0)
            self.assertIsInstance(result.exception, ValueError)
            self.assertIn('Finnhub tokens not found in the environment variable $FINNHUBTOKEN', str(result.exception))

    @patch('finsim.cli.FinnHubStockReader')
    def test_success_with_useenvtoken_json(self, mock_reader_cls):
        mock_reader = MagicMock()
        mock_reader.get_all_US_symbols.return_value = self.sample_symbols
        mock_reader_cls.return_value = mock_reader

        with tempfile.TemporaryDirectory() as tmpdir:
            out_file = os.path.join(tmpdir, 'symbols.json')
            result = self.runner.invoke(
                retrieve_stock_symbols,
                [out_file, '--useenvtoken'],
                env={'FINNHUBTOKEN': 'dummy_env_token'}
            )
            self.assertEqual(result.exit_code, 0)
            mock_reader_cls.assert_called_once_with('dummy_env_token')
            self.assertTrue(os.path.exists(out_file))

            df = pd.read_json(out_file)
            self.assertEqual(len(df), len(self.sample_symbols))

    @patch('finsim.cli.FinnHubStockReader')
    def test_success_with_finnhubtokenpath_and_shorten(self, mock_reader_cls):
        mock_reader = MagicMock()
        mock_reader.get_all_US_symbols.return_value = self.sample_symbols
        mock_reader_cls.return_value = mock_reader

        with tempfile.TemporaryDirectory() as tmpdir:
            token_file = os.path.join(tmpdir, 'token.txt')
            with open(token_file, 'w') as f:
                f.write('dummy_file_token\n')

            out_file = os.path.join(tmpdir, 'symbols.csv')
            result = self.runner.invoke(
                retrieve_stock_symbols,
                [out_file, '--finnhubtokenpath', token_file, '--shorten']
            )
            self.assertEqual(result.exit_code, 0)
            mock_reader_cls.assert_called_once_with('dummy_file_token')
            self.assertTrue(os.path.exists(out_file))

            df = pd.read_csv(out_file)
            # BRK.A has dot, TEST.PUB is PUBLIC, OTHER is BATS -> only AAPL and MSFT remain
            symbols = list(df['symbol'])
            self.assertEqual(symbols, ['AAPL', 'MSFT'])

    @patch('finsim.cli.FinnHubStockReader')
    def test_nonexistent_directory(self, mock_reader_cls):
        mock_reader = MagicMock()
        mock_reader.get_all_US_symbols.return_value = self.sample_symbols
        mock_reader_cls.return_value = mock_reader

        out_file = '/nonexistent/path/for/symbols.json'
        result = self.runner.invoke(
            retrieve_stock_symbols,
            [out_file, '--useenvtoken'],
            env={'FINNHUBTOKEN': 'dummy_env_token'}
        )
        self.assertNotEqual(result.exit_code, 0)
        self.assertIsInstance(result.exception, FileNotFoundError)

    @patch('finsim.cli.FinnHubStockReader')
    def test_unrecognized_extension(self, mock_reader_cls):
        mock_reader = MagicMock()
        mock_reader.get_all_US_symbols.return_value = self.sample_symbols
        mock_reader_cls.return_value = mock_reader

        with tempfile.TemporaryDirectory() as tmpdir:
            out_file = os.path.join(tmpdir, 'symbols.txt')
            result = self.runner.invoke(
                retrieve_stock_symbols,
                [out_file, '--useenvtoken'],
                env={'FINNHUBTOKEN': 'dummy_env_token'}
            )
            self.assertNotEqual(result.exit_code, 0)
            self.assertIsInstance(result.exception, IOError)


if __name__ == '__main__':
    unittest.main()
