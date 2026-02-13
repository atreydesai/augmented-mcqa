import unittest
from unittest.mock import MagicMock, patch
from pathlib import Path
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from data.hub_utils import push_dataset_to_hub

class TestHubPush(unittest.TestCase):
    
    @patch('data.hub_utils.HF_SKIP_PUSH', True)
    @patch('data.hub_utils.Dataset')
    def test_push_skipped_when_flag_is_true(self, mock_dataset):
        # Setup
        entries = [{"a": 1}]
        repo_id = "test/repo"
        
        # Execute
        result = push_dataset_to_hub(entries, repo_id=repo_id)
        
        # Verify
        self.assertIsNone(result)
        mock_dataset.from_list.assert_not_called()

    @patch('data.hub_utils.HF_SKIP_PUSH', False)
    @patch('data.hub_utils.HF_TOKEN', '')
    @patch('data.hub_utils.Dataset')
    def test_push_skipped_when_token_missing(self, mock_dataset):
        # Setup
        entries = [{"a": 1}]
        repo_id = "test/repo"
        
        # Execute
        result = push_dataset_to_hub(entries, repo_id=repo_id)
        
        # Verify
        self.assertIsNone(result)
        mock_dataset.from_list.assert_not_called()

    @patch('data.hub_utils.HF_SKIP_PUSH', False)
    @patch('data.hub_utils.HF_TOKEN', 'fake_token')
    @patch('data.hub_utils.Dataset')
    def test_push_success(self, mock_dataset_class):
        # Setup
        entries = [{"a": 1}]
        repo_id = "test/repo"
        mock_dataset_instance = MagicMock()
        mock_dataset_class.from_list.return_value = mock_dataset_instance
        
        # Execute
        result = push_dataset_to_hub(entries, repo_id=repo_id)
        
        # Verify
        self.assertEqual(result, f"https://huggingface.co/datasets/{repo_id}")
        mock_dataset_class.from_list.assert_called_once_with(entries)
        mock_dataset_instance.push_to_hub.assert_called_once_with(
            repo_id=repo_id,
            token='fake_token',
            private=False
        )

if __name__ == '__main__':
    unittest.main()
