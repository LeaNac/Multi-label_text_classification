from src.data.feature_engineering import tokenize


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
