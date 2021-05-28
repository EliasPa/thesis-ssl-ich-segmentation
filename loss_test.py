import unittest
from networks.unet2D.loss import dice_loss, intersection_over_union_loss
import torch

"""
Unit tests for Dice and IoU loss functions.
Run the unit tests with `python3 loss_test.py -v`.
"""

class DiceTest(unittest.TestCase):

    def test_when_half_correct_returns_half(self):
        # setup
        A = torch.tensor(
            [
                [
                    [
                        [0,1],
                        [0,1]
                    ],
                    [
                        [1,0],
                        [1,0.0]
                    ]
                ]
            ]
        )

        B = torch.tensor(
            [
                [
                    [1,1],
                    [1,1]
                ]
            ]
        )

        # act
        loss = dice_loss(A, B).item()

        # assert
        self.assertAlmostEqual(loss, 0.5, 4)

    def test_when_completely_different_returns_one(self):
        # setup
        A = torch.tensor(
            [
                [
                    [
                        [1,1],
                        [1,1]
                    ],
                    [
                        [0,0],
                        [0,0.0]
                    ]
                ]
            ]
        )

        B = torch.tensor(
            [
                [
                    [1,1],
                    [1,1]
                ]
            ]
        )

        # act
        loss = dice_loss(A, B).item()

        # assert
        self.assertAlmostEqual(loss, 1, 4)

    def test_when_same_returns_zero(self):
        # setup
        A = torch.tensor(
            [
                [
                    [
                        [0,0],
                        [0,0]
                    ],
                    [
                        [1,1],
                        [1,1.0]
                    ]
                ]
            ]
        )

        B = torch.tensor(
            [
                [
                    [1,1],
                    [1,1]
                ]
            ]
        )

        # act
        loss = dice_loss(A, B).item()

        # assert
        self.assertAlmostEqual(loss, 0, 4)

    def test_when_arbitrary_amount_correct_returns_correct_result(self):
        # setup
        A = torch.tensor(
            [
                [
                    [
                        [0,1],
                        [1,1]
                    ],
                    [
                        [1,0],
                        [0,0.0]
                    ]
                ]
            ]
        )

        B = torch.tensor(
            [
                [
                    [1,1],
                    [0,0]
                ]
            ]
        )

        # act
        loss = dice_loss(A, B).item()

        # assert
        # intersection = 3
        # union + intersection = 2 + 3 + 3 = 8
        # dice = 2*3 / 8 = 6 / 8 = 3 / 4 = 0.75 --> loss = 0.25
        self.assertAlmostEqual(loss, 0.25, 4)

    def test_when_arbitrary_amount_correct_and_multiclass_returns_correct_result(self):
        # setup
        A = torch.tensor(
            [
                [
                    [
                        [0,1],
                        [0,1]
                    ],
                    [
                        [0,0],
                        [1,0.0]
                    ],
                    [
                        [1,0],
                        [0,0.0]
                    ]
                ]
            ]
        )

        B = torch.tensor(
            [
                [
                    [2,1],
                    [1,0]
                ]
            ]
        )

        # act
        loss = dice_loss(A, B).item()

        # assert
        # intersection = 1 + 1 + 1 = 3
        # union + intersection = 3 + 3 + 2 = 8
        # dice = 2*3 / 8 = 6 / 8 = 3 / 4 = 0.75 --> loss = 0.25
        self.assertAlmostEqual(loss, 0.25, 4)

    def test_when_arbitrary_amount_correct_and_multiclass_returns_correct_result_2(self):
        # setup
        A = torch.tensor(
            [
                [
                    [
                        [1,1],
                        [0,1]
                    ],
                    [
                        [0,0],
                        [1,0.0]
                    ],
                    [
                        [0,0],
                        [0,0.0]
                    ]
                ]
            ]
        )

        B = torch.tensor(
            [
                [
                    [2,1],
                    [1,0]
                ]
            ]
        )

        # act
        loss = dice_loss(A, B).item()

        # assert
        # intersection = 1 + 1 + 0 = 2
        # union + intersection = 4 + 3 + 1 = 8
        # dice = 2*2 / 8 = 4 / 8 = 1 / 2 = 0.5 --> loss = 0.5
        self.assertAlmostEqual(loss, 0.5, 4)

class IoUTest(unittest.TestCase):

    def test_when_half_correct_returns_correct_result(self):
        # setup
        A = torch.tensor(
            [
                [
                    [
                        [0,1],
                        [0,1]
                    ],
                    [
                        [1,0],
                        [1,0.0]
                    ]
                ]
            ]
        )

        B = torch.tensor(
            [
                [
                    [1,1],
                    [1,1]
                ]
            ]
        )

        # act
        loss = intersection_over_union_loss(A, B).item()

        # assert
        self.assertAlmostEqual(loss, 0.66666, 4)

    def test_when_arbitrary_amount_correct_returns_correct_result(self):
        # setup
        A = torch.tensor(
            [
                [
                    [
                        [0,1],
                        [1,1]
                    ],
                    [
                        [1,0],
                        [0,0.0]
                    ]
                ]
            ]
        )

        B = torch.tensor(
            [
                [
                    [1,1],
                    [0,0]
                ]
            ]
        )

        # act
        loss = intersection_over_union_loss(A, B).item()

        # assert
        # intersection = 3
        # union = 2 + 3 = 5
        # iou = 3 / 5 = 0.6 --> loss = 1 - iou = 0.4
        self.assertAlmostEqual(loss, 0.4, 4)

    def test_when_completely_different_returns_one(self):
        # setup
        A = torch.tensor(
            [
                [
                    [
                        [1,1],
                        [1,1]
                    ],
                    [
                        [0,0],
                        [0,0.0]
                    ]
                ]
            ]
        )

        B = torch.tensor(
            [
                [
                    [1,1],
                    [1,1]
                ]
            ]
        )

        # act
        loss = intersection_over_union_loss(A, B).item()

        # assert
        self.assertAlmostEqual(loss, 1, 4)

    def test_when_same_returns_zero(self):
        # setup
        A = torch.tensor(
            [
                [
                    [
                        [0,0],
                        [0,0]
                    ],
                    [
                        [1,1],
                        [1,1.0]
                    ]
                ]
            ]
        )

        B = torch.tensor(
            [
                [
                    [1,1],
                    [1,1]
                ]
            ]
        )

        # act
        loss = intersection_over_union_loss(A, B).item()

        # assert
        self.assertAlmostEqual(loss, 0, 4)


    def test_when_arbitrary_amount_correct_and_multiclass_returns_correct_result(self):
        # setup
        A = torch.tensor(
            [
                [
                    [
                        [0,1],
                        [0,1]
                    ],
                    [
                        [0,0],
                        [1,0.0]
                    ],
                    [
                        [1,0],
                        [0,0.0]
                    ]
                ]
            ]
        )

        B = torch.tensor(
            [
                [
                    [2,1],
                    [1,0]
                ]
            ]
        )

        # act
        loss = intersection_over_union_loss(A, B).item()

        # assert
        # intersection = 1 + 1 + 1 = 3
        # union = 2 + 2 + 1 = 5
        # iou = 3 / 5 = 0.6 --> loss = 0.4
        self.assertAlmostEqual(loss, 0.4, 4)

    def test_when_arbitrary_amount_correct_and_multiclass_returns_correct_result_2(self):
        # setup
        A = torch.tensor(
            [
                [
                    [
                        [1,1],
                        [0,1]
                    ],
                    [
                        [0,0],
                        [1,0.0]
                    ],
                    [
                        [0,0],
                        [0,0.0]
                    ]
                ]
            ]
        )

        B = torch.tensor(
            [
                [
                    [2,1],
                    [1,0]
                ]
            ]
        )

        # act
        loss = intersection_over_union_loss(A, B).item()

        # assert
        # intersection = 1 + 1 + 0 = 2
        # union = 3 + 2 + 1 = 6
        # dice = 2 / 6 = 1 / 3 = 0.3333 --> loss = 0.66666
        self.assertAlmostEqual(loss, 0.666666, 4)

if __name__ == '__main__':
    unittest.main()
