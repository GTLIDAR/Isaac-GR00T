from gr00t.configs.data.embodiment_configs import register_modality_config
from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.data.types import (
    ModalityConfig,
    ActionConfig,
    ActionRepresentation,
    ActionType,
    ActionFormat,
)


digit_v3_config = {
    "video": ModalityConfig(
        delta_indices=[0],
        modality_keys=["rgb"],
    ),
    "state": ModalityConfig(
        delta_indices=[0],
        modality_keys=["lin_vel_in_body", "ang_vel_in_body", "qpos", "qvel"],
    ),
    "torque": ModalityConfig(
        delta_indices=[0],
        modality_keys=["upper_body_torque"],
    ),
    "action": ModalityConfig(
        delta_indices=list(range(0, 16)),
        modality_keys=["ref_ee_pos_in_body", "ref_lhand_contact_force", "ref_rhand_contact_force"],
        action_configs=[
            ActionConfig(
                rep=ActionRepresentation.ABSOLUTE,
                type=ActionType.EEF,
                format=ActionFormat.DEFAULT,
            ),
        ],
    ),
    "language": ModalityConfig(
        delta_indices=[0],
        modality_keys=[
            "annotation.human.goal",
            # optionally add paraphrases:
            # "annotation.human.paraphrase_0",
            # "annotation.human.paraphrase_1",
        ],
    ),
}

register_modality_config(digit_v3_config, embodiment_tag=EmbodimentTag.NEW_EMBODIMENT)
