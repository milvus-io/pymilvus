import pytest
from pymilvus.client.check import check_pass_param
from pymilvus.client.utils import generate_timestamp


class TestCheckPassParam:
    def test_check_pass_param_valid(self):
        a = [[i * j for i in range(20)] for j in range(20)]
        check_pass_param(search_data=a)

        import numpy as np
        a = np.float32([[1, 2, 3, 4], [1, 2, 3, 4]])
        check_pass_param(search_data=a)

    def test_check_param_invalid(self):
        with pytest.raises(Exception):
            a = {[i * j for i in range(20) for j in range(20)]}
            check_pass_param(search_data=a)

        with pytest.raises(Exception):
            a = [{i * j for i in range(40)} for j in range(40)]
            check_pass_param(search_data=a)


class TestGenTS:
    def test_gen_timestamp(self):
        t = generate_timestamp(426152581543231492, 1000)

        print(f"Generated ts: {t}")
