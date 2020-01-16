
import pytest
import dip.deep_fill.train_copy as deep_fill_train


@pytest.mark.parametrize('max_train_images', [1, 10, 100, 1000, 10000, 100000, 1000000, 999999999999])
def test_train_multi(max_train_images):
    log_dir = f'./logs/inpaint_{max_train_images}'
    deep_fill_train.train('./test/resources/inpaint_test.yml', max_train_images, log_dir)

