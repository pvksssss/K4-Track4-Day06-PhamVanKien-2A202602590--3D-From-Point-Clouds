"""Analytical checks independent of dataset appearance: python -m unittest src.test_projection."""
import unittest

import numpy as np

from starter.kitti_io import KittiCalib, KittiObject
from starter.projection import cam_to_image, velo_to_cam, box3d_corners_cam
from src.calibration_qa import object_members


class ProjectionTests(unittest.TestCase):
    def test_lidar_axes_and_translation(self):
        transform = np.array([[0, -1, 0, 1], [0, 0, -1, 2], [1, 0, 0, 3]], dtype=float)
        calib = KittiCalib(np.zeros((3, 4)), np.eye(3), transform)
        np.testing.assert_allclose(velo_to_cam(np.array([[10, 0, 0], [5, 2, 1]]), calib),
                                   [[1, 2, 13], [-1, 1, 8]])

    def test_rectification_is_applied(self):
        rectify = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=float)
        calib = KittiCalib(np.zeros((3, 4)), rectify, np.eye(4)[:3])
        np.testing.assert_allclose(velo_to_cam(np.array([[2, 3, 4]]), calib), [[-3, 2, 4]])

    def test_projection_and_original_mask(self):
        P = np.array([[10, 0, 5, 2], [0, 10, 5, 0], [0, 0, 1, 0]], dtype=float)
        points = np.array([[0, 0, 2], [0, 0, -1], [np.nan, 0, 2], [0, np.inf, 2],
                           [2, 0, 2], [-1.2, -1, 2], [0, 0, .1]])
        uv, depth, mask = cam_to_image(points, P, (10, 10, 3))
        np.testing.assert_array_equal(mask, [True, False, False, False, False, True, False])
        np.testing.assert_allclose(uv, [[6, 5], [0, 0]], atol=1e-12)
        np.testing.assert_allclose(depth, [2, 2])

    def test_empty_and_zero_projective_denominator(self):
        P = np.eye(3, 4)
        uv, depth, mask = cam_to_image(np.empty((0, 3)), P, (10, 10))
        self.assertEqual(uv.shape, (0, 2))
        self.assertEqual(depth.shape, (0,))
        self.assertEqual(mask.shape, (0,))
        P[2] = 0
        uv, _, mask = cam_to_image(np.array([[1, 1, 2]]), P, (10, 10))
        self.assertFalse(mask.any())
        self.assertEqual(len(uv), 0)

    def test_rotated_object_membership(self):
        obj = KittiObject("Car", 0, 0, 0, np.zeros(4), np.array([2, 2, 4]),
                          np.array([1, 2, 10]), np.pi / 2)
        corners = box3d_corners_cam(obj)
        self.assertTrue(object_members(corners, obj).all())
        points = np.array([[1, 1, 10], [1, 3, 10], [1, 1, 14], [np.nan, 1, 10]])
        np.testing.assert_array_equal(object_members(points, obj), [True, False, False, False])


if __name__ == "__main__":
    unittest.main()
