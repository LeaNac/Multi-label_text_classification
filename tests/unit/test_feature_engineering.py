import numpy as np
import pandas as pd
import scipy.sparse
from numpy.testing import assert_array_equal
from pandas.testing import assert_frame_equal

from src.data.feature_engineering import pr, data_preprocessing


class TestFeatureEngineering:
    def test_pr_should_return_expected_probabilities_with_X_as_dense_numpy_array(self):
        # Given
        X = np.array([[1, 0, 1],
                      [0, 1, 1]])

        y = np.array([0, 1])
        class_0 = 0
        class_1 = 1

        expected_proba_class_1 = np.array([1, 0.5, 1])
        expected_proba_class_2 = np.array([0.5, 1, 1])

        # When
        proba_class_1 = pr(X, y, class_0)
        proba_class_2 = pr(X, y, class_1)

        # Then
        assert_array_equal(proba_class_1, expected_proba_class_1)
        assert_array_equal(proba_class_2, expected_proba_class_2)

    def test_pr_should_return_expected_probabilities_with_X_as_sparse_matrix(self):
        # Given
        X = np.array([[1., 0., 1.],
                      [0., 1., 1.]])
        X_sparse = scipy.sparse.csr_matrix(X)

        y = np.array([0, 1])
        class_0 = 0
        class_1 = 1

        expected_proba_class_1 = np.array([1, 0.5, 1])
        expected_proba_class_2 = np.array([0.5, 1, 1])

        # When
        proba_class_1 = pr(X_sparse, y, class_0)
        proba_class_2 = pr(X_sparse, y, class_1)

        # Then
        np.array_equiv(proba_class_1, expected_proba_class_1)
        np.array_equiv(proba_class_2, expected_proba_class_2)

    def test_fixture(self, dataset_1):
        print(dataset_1)
        assert True

    def test_data_preprocessing_should_return_df_with_none_colunm(self):
        # Given
        df = pd.DataFrame([['', 0, 0, 1], ['', 0, 0, 0]], columns=['comment_text', 'c1', 'c2', 'c3'])
        expected = pd.DataFrame([['', 0, 0, 1, 0], ['', 0, 0, 0, 1]],
                                columns=['comment_text', 'c1', 'c2', 'c3', 'none'])
        # When
        actual, _ = data_preprocessing(df, df, ['c1', 'c2', 'c3'], 'comment_text')

        # Then
        assert_frame_equal(expected, actual)
