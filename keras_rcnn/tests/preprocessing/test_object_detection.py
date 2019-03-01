import keras_rcnn.preprocessing._object_detection
import numpy


def test_scale_shape():
    min_size = 200
    max_size = 300
    size     = (600, 1000)

    size, scale = keras_rcnn.preprocessing._object_detection.scale_size(size, min_size, max_size)

    expected = (180, 300)
    numpy.testing.assert_equal(size, expected)

    expected = 0.3
    assert numpy.isclose(scale, expected)
