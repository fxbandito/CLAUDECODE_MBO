"""
Sound Manager for MBO Trading Strategy Analyzer v5.
Handles all audio feedback for UI interactions.
"""
# pylint: disable=broad-exception-caught

import logging
import os
import platform
from concurrent.futures import ThreadPoolExecutor

# Optional imports for sound backends
try:
    import winsound
except ImportError:
    winsound = None

try:
    import pygame
except ImportError:
    pygame = None

# Determine which audio backend to use
SOUND_BACKEND = None  # pylint: disable=invalid-name

if platform.system() == "Windows" and winsound is not None:
    SOUND_BACKEND = "winsound"

if SOUND_BACKEND is None and pygame is not None:
    try:
        pygame.mixer.init()
        SOUND_BACKEND = "pygame"
    except Exception as e:
        logging.debug("pygame initialization failed: %s", e)

if SOUND_BACKEND is None:
    logging.debug("No sound backend available. Sound effects disabled.")


class SoundManager:
    """
    Manages sound effects for the application.

    Sound files:
        - app_start.wav      : On program start
        - app_close.wav      : On program close
        - tab_switch.wav     : On tab switch
        - button_click.wav   : On any button press
        - toggle_switch.wav  : Toggle buttons (hu-en, dark mode, etc.)
        - checkbox_on.wav    : Checkbox checked
        - checkbox_off.wav   : Checkbox unchecked
        - model_start.wav    : Model run start
        - model_complete.wav : Model run complete
    """

    _instance = None

    def __new__(cls):
        """Singleton pattern."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if getattr(self, "_initialized", False):
            return

        self._initialized = True
        self.enabled: bool = False  # Default: muted
        self.volume = 0.5
        self.sounds = {}
        self.backend = SOUND_BACKEND

        self._executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="sound_")
        self.sounds_dir = os.path.join(os.path.dirname(__file__), "sounds")

        self.sound_files = {
            "app_start": "app_start.wav",
            "app_close": "app_close.wav",
            "tab_switch": "tab_switch.wav",
            "button_click": "button_click.wav",
            "toggle_switch": "toggle_switch.wav",
            "checkbox_on": "checkbox_on.wav",
            "checkbox_off": "checkbox_off.wav",
            "model_start": "model_start.wav",
            "model_complete": "model_complete.wav",
        }

        self._load_sounds()

    def _load_sounds(self):
        """Load all sound files into memory (pygame only)."""
        if self.backend != "pygame" or pygame is None:
            return

        for name, filename in self.sound_files.items():
            filepath = os.path.join(self.sounds_dir, filename)
            try:
                if os.path.exists(filepath):
                    self.sounds[name] = pygame.mixer.Sound(filepath)
                    self.sounds[name].set_volume(self.volume)
            except Exception as e:
                logging.debug("Failed to load sound %s: %s", name, e)

    def play(self, sound_name: str):
        """Play a sound effect."""
        if not self.enabled or self.backend is None:
            return

        if sound_name not in self.sound_files:
            return

        filepath = os.path.join(self.sounds_dir, self.sound_files[sound_name])
        if not os.path.exists(filepath):
            return

        try:
            self._executor.submit(self._play_sound, sound_name, filepath)
        except Exception:
            pass

    def _play_sound(self, sound_name: str, filepath: str):
        """Internal method to play sound (called from thread)."""
        try:
            if self.backend == "winsound" and winsound is not None:
                winsound.PlaySound(filepath, winsound.SND_FILENAME)
            elif self.backend == "pygame" and pygame is not None:
                if sound_name in self.sounds:
                    self.sounds[sound_name].play()
        except Exception:
            pass

    def set_enabled(self, enabled: bool):
        """Enable or disable sound effects."""
        self.enabled = enabled

    def toggle(self) -> bool:
        """Toggle sound effects on/off. Returns new state."""
        self.enabled = not self.enabled
        return self.enabled

    def set_volume(self, volume: float):
        """Set volume for all sounds (0.0 - 1.0)."""
        self.volume = max(0.0, min(1.0, volume))

        if self.backend != "pygame":
            return

        for sound in self.sounds.values():
            try:
                sound.set_volume(self.volume)
            except Exception:
                pass

    def is_available(self) -> bool:
        """Check if sound system is available."""
        return self.backend is not None

    def shutdown(self):
        """Shutdown the thread pool."""
        if hasattr(self, "_executor") and self._executor:
            self._executor.shutdown(wait=False)

    # Convenience methods for common sound effects
    def play_app_start(self):
        """Play application start sound."""
        self.play("app_start")

    def play_app_close(self):
        """Play application close sound."""
        self.play("app_close")

    def play_tab_switch(self):
        """Play tab switch sound."""
        self.play("tab_switch")

    def play_button_click(self):
        """Play button click sound."""
        self.play("button_click")

    def play_toggle_switch(self):
        """Play toggle switch sound."""
        self.play("toggle_switch")

    def play_checkbox_on(self):
        """Play checkbox checked sound."""
        self.play("checkbox_on")

    def play_checkbox_off(self):
        """Play checkbox unchecked sound."""
        self.play("checkbox_off")

    def play_model_start(self):
        """Play model run start sound."""
        self.play("model_start")

    def play_model_complete(self):
        """Play model run complete sound."""
        self.play("model_complete")


# Global instance
_SOUND_MANAGER = None


def get_sound_manager() -> SoundManager:
    """Get the global SoundManager instance."""
    global _SOUND_MANAGER
    if _SOUND_MANAGER is None:
        _SOUND_MANAGER = SoundManager()
    return _SOUND_MANAGER
