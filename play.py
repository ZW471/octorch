"""
Interactive CHIP-8 emulator with simple score detection (PyTorch port of Octax's play.py).

Usage:
    uv run python play.py "roms/Brix [Andreas Gustafsson, 1990].ch8"
"""

import sys
import time

import numpy as np
import pygame

from octorch import create_state, load_rom
from octorch.interpreter import ScalarChip8

# Modern key mapping
KEY_MAP = {
    pygame.K_1: 0x1, pygame.K_2: 0x2, pygame.K_3: 0x3, pygame.K_4: 0x4,
    pygame.K_5: 0x5, pygame.K_6: 0x6, pygame.K_7: 0x7, pygame.K_8: 0x8,
    pygame.K_9: 0x9, pygame.K_0: 0x0,
    pygame.K_UP: 0x2, pygame.K_DOWN: 0x8, pygame.K_LEFT: 0x4, pygame.K_RIGHT: 0x6,
    pygame.K_w: 0x2, pygame.K_s: 0x8, pygame.K_a: 0x4, pygame.K_d: 0x6,
    pygame.K_SPACE: 0x5,
    pygame.K_q: 0xA, pygame.K_e: 0xB, pygame.K_t: 0xC,
    pygame.K_y: 0xD, pygame.K_u: 0xE, pygame.K_i: 0xF,
}


class SimpleDetector:
    """Lightweight register tracking for score-register discovery."""

    def __init__(self):
        self.register_history = [[] for _ in range(16)]
        self.last_values = [0] * 16
        self.bcd_registers = set()
        self.last_print_time = 0

    def detect_bcd(self, machine: ScalarChip8, instruction: int):
        """Detect BCD operations - strongest score indicator."""
        if (instruction & 0xF0FF) == 0xF033:
            register = (instruction & 0x0F00) >> 8
            self.bcd_registers.add(register)
            print(f"🎯 BCD! V{register:X} = {int(machine.V[register])} -> MEM[0x{machine.I:03X}]")

    def track_changes(self, machine: ScalarChip8):
        for reg in range(16):
            current = int(machine.V[reg])
            if current != self.last_values[reg]:
                self.register_history[reg].append(current)
                if len(self.register_history[reg]) > 5:
                    self.register_history[reg].pop(0)
                self.last_values[reg] = current

    def print_all_registers(self, machine: ScalarChip8, instruction_count: int):
        now = time.time()
        if now - self.last_print_time < 30.0:
            return
        self.last_print_time = now
        print(f"\n📊 ALL REGISTERS (Instruction #{instruction_count}):")
        for reg in range(16):
            changes = self.register_history[reg]
            if len(changes) >= 2:
                hist = f" [{' -> '.join(str(v) for v in changes)}]"
                inc = sum(1 for i in range(1, len(changes)) if changes[i] > changes[i - 1])
                dec = sum(1 for i in range(1, len(changes)) if changes[i] < changes[i - 1])
                trend = "📈" if inc >= len(changes) // 2 else "📉" if dec >= len(changes) // 2 else "📊"
            elif len(changes) == 1:
                hist, trend = f" [changed to {changes[0]}]", "📊"
            else:
                hist, trend = " [no changes]", "📊"
            bcd_mark = " 🎯" if reg in self.bcd_registers else ""
            print(f"  V{reg:X}: {int(machine.V[reg]):3d}{hist} {trend}{bcd_mark}")
        if self.bcd_registers:
            print(f"🔢 BCD Registers: {', '.join(f'V{r:X}' for r in sorted(self.bcd_registers))}")
        print("-" * 60)


def draw_overlay_text(surface, text_lines, position, font, bg_color=(0, 0, 0), text_color=(255, 255, 255), alpha=120):
    if not text_lines:
        return
    line_height = font.get_height()
    max_width = max(font.size(line)[0] for line in text_lines)
    overlay = pygame.Surface((max_width + 16, len(text_lines) * line_height + 8))
    overlay.set_alpha(alpha)
    overlay.fill(bg_color)
    surface.blit(overlay, position)
    x, y = position
    for i, line in enumerate(text_lines):
        surface.blit(font.render(line, True, text_color), (x + 8, y + 4 + i * line_height))


def run_emulator(rom_filename, modern_mode=True, scale=8, ipf=17):
    """Main emulator loop - authentic CHIP-8 settings."""
    pygame.init()
    screen = pygame.display.set_mode((64 * scale, 32 * scale))
    pygame.display.set_caption("CHIP-8 Score Detective")
    clock = pygame.time.Clock()

    def fresh_machine():
        state = create_state(0, device="cpu", modern_mode=modern_mode)
        return ScalarChip8(load_rom(state, rom_filename))

    try:
        machine = fresh_machine()
        print(f"✅ Loaded: {rom_filename}")
    except Exception as e:
        print(f"❌ Error: {e}")
        return

    keypad_state = np.zeros(16, dtype=bool)
    instruction_count = 0
    running, paused, show_debug = True, False, True
    detector = SimpleDetector()
    snapshot_timer = 0
    start_time = time.time()
    frame_count, fps_start_time, current_fps = 0, time.time(), 60

    print("🎮 Controls: ESC=Quit, P=Pause, R=Reset, ±=Speed, D=Debug")
    print("🎯 Watching for BCD operations and register patterns...")

    while running:
        clock.tick(60)
        frame_count += 1
        if time.time() - fps_start_time >= 1.0:
            current_fps = frame_count / (time.time() - fps_start_time)
            frame_count, fps_start_time = 0, time.time()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_p:
                    paused = not paused
                elif event.key == pygame.K_d:
                    show_debug = not show_debug
                elif event.key == pygame.K_r:
                    machine = fresh_machine()
                    keypad_state[:] = False
                    instruction_count = 0
                    detector = SimpleDetector()
                    print("🔄 Reset")
                elif event.key == pygame.K_EQUALS:
                    ipf = min(100, ipf + 3)
                    print(f"⚡ Speed: {ipf} IPF")
                elif event.key == pygame.K_MINUS:
                    ipf = max(3, ipf - 3)
                    print(f"🐌 Speed: {ipf} IPF")
                elif event.key in KEY_MAP:
                    keypad_state[KEY_MAP[event.key]] = True
            elif event.type == pygame.KEYUP:
                if event.key in KEY_MAP:
                    keypad_state[KEY_MAP[event.key]] = False

        if not paused:
            machine.keypad[:] = keypad_state
            if machine.delay_timer > 0:
                machine.delay_timer -= 1
            if machine.sound_timer > 0:
                machine.sound_timer -= 1
            for i in range(ipf):
                instruction = machine.fetch()
                if i % 5 == 0:
                    detector.detect_bcd(machine, instruction)
                machine.execute(instruction)
                instruction_count += 1
            detector.track_changes(machine)
            snapshot_timer += ipf
            if snapshot_timer >= 1800:
                detector.print_all_registers(machine, instruction_count)
                snapshot_timer = 0

        screen.fill((0, 0, 0))
        for x, y in zip(*np.nonzero(machine.display)):
            pygame.draw.rect(screen, (0, 255, 0), pygame.Rect(int(x) * scale, int(y) * scale, scale, scale))

        if show_debug:
            font_small, font_tiny = pygame.font.Font(None, 18), pygame.font.Font(None, 16)
            runtime = time.time() - start_time
            ips = instruction_count / runtime if runtime > 0 else 0
            draw_overlay_text(screen, [
                f"PC: 0x{machine.pc:03X}", f"I: 0x{machine.I:03X}", f"Instructions: {instruction_count}",
                f"CPU: {ips:.0f} Hz (target: ~600 Hz)", f"IPF: {ipf}", f"FPS: {current_fps:.1f} (target: 60)",
                f"Status: {'PAUSED' if paused else 'RUNNING'}",
            ], (5, 5), font_small, alpha=100)
            draw_overlay_text(screen, [f"Delay: {machine.delay_timer}", f"Sound: {machine.sound_timer}"],
                              (64 * scale - 80, 5), font_small, alpha=100)
            reg_lines = []
            for i in range(0, 16, 4):
                parts = []
                for j in range(i, i + 4):
                    v = f"{int(machine.V[j]):02X}"
                    parts.append(f"*V{j:X}:{v}*" if j in detector.bcd_registers else f"V{j:X}:{v}")
                reg_lines.append(" ".join(parts))
            draw_overlay_text(screen, reg_lines, (5, 32 * scale - 80), font_tiny, alpha=80)
            if detector.bcd_registers:
                draw_overlay_text(screen, ["BCD Registers:", ", ".join(f"V{r:X}" for r in sorted(detector.bcd_registers))],
                                  (64 * scale - 120, 80), font_tiny, text_color=(255, 255, 0), alpha=120)
            pressed = [f"{i:X}" for i in range(16) if keypad_state[i]]
            if pressed:
                draw_overlay_text(screen, ["Keys: " + " ".join(pressed)], (64 * scale - 100, 32 * scale - 25), font_tiny, alpha=150)
        elif paused:
            screen.blit(pygame.font.Font(None, 24).render("PAUSED - P to resume", True, (255, 255, 0)), (10, 10))

        pygame.display.flip()
    pygame.quit()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    run_emulator(sys.argv[1], modern_mode="--legacy" not in sys.argv)
