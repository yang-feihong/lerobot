import torch

from lerobot.policies.pi05.processor_pi05 import Pi05ActionRepresentationProcessorStep


def test_terminal_static_right_padding_is_unmasked():
    actions = torch.zeros(2, 5, 25)
    actions[:, :, 3] = 1.0
    padding = torch.tensor(
        [
            [False, False, True, True, True],
            [False, False, True, True, True],
        ]
    )
    # The second sample ends while B2 is still moving and must remain masked.
    actions[1, 1, 0] = 0.2
    result = Pi05ActionRepresentationProcessorStep._terminal_static_padding_mask(actions, padding)
    assert result[0].tolist() == [False, False, False, False, False]
    assert result[1].tolist() == [False, False, True, True, True]


def test_non_trailing_padding_is_never_unmasked():
    actions = torch.zeros(1, 4, 25)
    actions[:, :, 3] = 1.0
    padding = torch.tensor([[False, True, False, True]])
    result = Pi05ActionRepresentationProcessorStep._terminal_static_padding_mask(actions, padding)
    assert torch.equal(result, padding)
