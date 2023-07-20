import unittest

import pandas as pd
from taxonomyml.main import get_data


class TestData(unittest.TestCase):
    def test_get_data(self):
        # Test with a pandas dataframe
        data = pd.DataFrame({"text": ["hello", "world"], "search_volume": [100, 200]})
        result = get_data(
            data=data, text_column="text", search_volume_column="search_volume"
        )
        self.assertEqual(result.shape, (2, 2))

        # Test with Google Search Console data
        data = "sc-domain:example.com"
        result = get_data(data=data, days=7, brand_terms=["example"], limit_queries=10)
        self.assertGreater(result.shape[0], 0)


if __name__ == "__main__":
    unittest.main()
