import json
import pygame
from gesture_engine import GestureEngine
from scene_manager import SceneManager
from audio_manager import AudioManager


def main():
    with open("config.json") as f:
        config = json.load(f)

    pygame.init()
    screen = pygame.display.set_mode(tuple(config["resolution"]), pygame.FULLSCREEN)
    pygame.display.set_caption("Black Heritage Reclaimed")
    clock = pygame.time.Clock()

    audio = AudioManager(config)
    gesture = GestureEngine(config)
    scene = SceneManager(config, screen, audio)

    scene.load(config["start_scene"])

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                running = False

        detected = gesture.update()
        if detected:
            scene.handle_gesture(detected)

        scene.update()
        scene.draw()
        pygame.display.flip()
        clock.tick(60)

    gesture.release()
    pygame.quit()


if __name__ == "__main__":
    main()
