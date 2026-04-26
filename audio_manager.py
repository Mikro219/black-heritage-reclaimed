import pygame


class AudioManager:
    def __init__(self, config):
        pygame.mixer.init()
        self._offset_ms = config.get("projector_audio_offset_ms", 0)
        self._paused = False

        self._narration_channel = pygame.mixer.Channel(0)
        self._ambience_channel = pygame.mixer.Channel(1)
        self._sfx_channel = pygame.mixer.Channel(2)

    def play_scene(self, metadata):
        self._narration_channel.stop()
        self._ambience_channel.stop()

        audio = metadata.get("audio", {})

        if "narration" in audio:
            sound = pygame.mixer.Sound(audio["narration"])
            self._narration_channel.play(sound)

        if "ambience" in audio:
            sound = pygame.mixer.Sound(audio["ambience"])
            self._ambience_channel.play(sound, loops=-1)

    def play_sfx(self, path):
        sound = pygame.mixer.Sound(path)
        self._sfx_channel.play(sound)

    def toggle_pause(self):
        if self._paused:
            pygame.mixer.unpause()
        else:
            pygame.mixer.pause()
        self._paused = not self._paused

    def stop_all(self):
        pygame.mixer.stop()
