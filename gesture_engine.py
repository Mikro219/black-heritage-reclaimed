import cv2
import mediapipe as mp


class GestureEngine:
    def __init__(self, config):
        self._cap = cv2.VideoCapture(config["camera_index"])
        self._mp_hands = mp.solutions.hands
        self._hands = self._mp_hands.Hands(
            min_detection_confidence=config["min_detection_confidence"],
            min_tracking_confidence=config["min_tracking_confidence"],
        )
        self._cooldown_ms = config["gesture_cooldown_ms"]
        self._confirm_frames = config["gesture_confirmation_frames"]
        self._debug = config.get("debug_overlay", False)

        self._last_gesture_tick = 0
        self._candidate = None
        self._candidate_count = 0

        # Wrist x-position from previous frame, used to compute swipe direction.
        self._prev_wrist_x = None

    def update(self):
        ret, frame = self._cap.read()
        if not ret:
            return None

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self._hands.process(rgb)

        gesture = self._classify(results, frame.shape)

        if self._debug:
            cv2.imshow("debug", frame)
            cv2.waitKey(1)

        return self._debounce(gesture)

    def _classify(self, results, shape):
        if not results.multi_hand_landmarks:
            self._prev_wrist_x = None
            return None

        landmarks = results.multi_hand_landmarks[0].landmark
        wrist_x = landmarks[0].x

        gesture = None
        if self._prev_wrist_x is not None:
            delta = wrist_x - self._prev_wrist_x
            if delta > 0.08:
                gesture = "swipe_right"
            elif delta < -0.08:
                gesture = "swipe_left"

        # Open-palm hold: all fingertips above their MCP joints.
        tips = [4, 8, 12, 16, 20]
        mcps = [2, 5, 9, 13, 17]
        if all(landmarks[t].y < landmarks[m].y for t, m in zip(tips, mcps)):
            gesture = "open_palm"

        self._prev_wrist_x = wrist_x
        return gesture

    def _debounce(self, gesture):
        now = pygame_ticks()
        if gesture != self._candidate:
            self._candidate = gesture
            self._candidate_count = 1
            return None

        self._candidate_count += 1
        if self._candidate_count < self._confirm_frames:
            return None
        if now - self._last_gesture_tick < self._cooldown_ms:
            return None

        self._last_gesture_tick = now
        self._candidate_count = 0
        return gesture

    def release(self):
        self._cap.release()
        self._hands.close()
        cv2.destroyAllWindows()


def pygame_ticks():
    import pygame
    return pygame.time.get_ticks()
