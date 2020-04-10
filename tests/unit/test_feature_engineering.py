import pandas as pd
from pandas.testing import assert_frame_equal

from src.data.feature_engineering import preprocess_train_df, tokenize


class TestFeatureEngineering:

    # def test_fixture(self, dataset_1):
    #     print(dataset_1)
    #     assert True
    #
    def test_tokenize_should_tokenize_string_into_a_cleaned_list_of_strings(self):
        # Given
        input_string = 'this is a sentence'
        expected_tokenized_string = ['this', 'is', 'a', 'sentence']

        # When
        actual_tokenized_string = tokenize(input_string)

        # Then
        assert actual_tokenized_string == expected_tokenized_string

    def test_data_preprocessing_should_return_df_with_none_column(self):
        # Given
        df = pd.DataFrame([['', 0, 0, 1], ['', 0, 0, 0]], columns=['comment_text', 'c1', 'c2', 'c3'])
        expected_df = pd.DataFrame([['', 0, 0, 1, 0], ['', 0, 0, 0, 1]],
                                   columns=['comment_text', 'c1', 'c2', 'c3', 'none'])
        # When
        actual_df = preprocess_train_df(df, ['c1', 'c2', 'c3'], 'comment_text')

        # Then
        assert_frame_equal(expected_df, actual_df)
