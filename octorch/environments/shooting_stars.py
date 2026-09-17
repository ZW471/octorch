import torch
import torch

from octorch import EmulatorState


rom_file = "Shooting Stars [Philip Baltzer, 1978].ch8"

def score_fn(state: EmulatorState) -> float:
    v0 = state.V[..., 0]
    return torch.where(v0 > 128, torch.zeros_like(v0), v0)


def terminated_fn(state: EmulatorState) -> bool:
    return False

action_set = [2, 8, 4, 6]

disable_delay = True

metadata = {
    "title": "Shooting Stars",
    "description": "Shooting Stars (1978), by Philip Baltzer",
    "authors": ["Philip Baltzer"],
    "release": "1978",
    "roms": {
      "443550abf646bc7f475ef0466f8e1232ec7474f3": {
        "file": "Shooting Stars [Philip Baltzer, 1978].ch8",
        "platforms": ["originalChip8"]
      }
    }
  }