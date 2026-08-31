# Black Heritage Reclaimed — How to Run the Experience

A simple guide for setting up and running the installation. No computer
knowledge needed beyond copying a folder and pressing a few keys.

---

## What this is

Black Heritage Reclaimed is an interactive story about the Underground
Railroad. Visitors stand in front of the screen and take part using **hand
gestures** (a camera watches them) and a few **spoken words** (a microphone
listens). One full journey takes about 7–8 minutes, and the experience
resets itself for the next visitor when it ends.

Everything lives in **one folder** called `BHR`. Inside it is the program,
`BHR.exe`, plus everything it needs. Nothing gets installed on the computer.

---

## One-time setup

1. **Copy the whole `BHR` folder** onto the computer — for example to
   `C:\BHR`. Don't copy just the `BHR.exe` file by itself; the folder must
   stay together.
2. **Plug in the camera and the microphone** (USB). Do this *before*
   starting the program.
3. Connect the **screen or projector** and the **speakers**.
4. **Run the preload once (recommended).** Double-click **`Prewarm cache (run once).bat`**
   and let it finish (it can take a while and needs plenty of free disk). This
   builds the speed-up files ahead of time so the very first show runs smoothly.
   You only ever need to do this once per computer.
5. That's it. No installation, no internet needed.

> Tip: avoid putting the folder inside OneDrive, Dropbox, or Google Drive
> folders — they will try to upload it and slow the computer down.

---

## Starting it up

1. Open the `BHR` folder and **double-click `BHR.exe`**.
2. A black text window opens first — **this is normal**, just leave it
   alone. The experience window appears after a few seconds.
3. **Camera check screen:** you'll see the live camera picture with a
   stick-figure drawn over anyone standing in view. Stand where a visitor
   would stand. When the stick figure turns **green** (head and feet both
   visible), the camera is aimed well. Adjust the camera if not.
   Press **Enter** (or say **"ready"**) to continue.
4. You'll now see the start screen, paused. When you're ready for the
   first visitor, press the **Space bar**.
5. A short **tutorial** plays first, teaching the visitor the basic
   gestures (raise hands, point, reach). It moves on by itself. Press
   **S** to skip it entirely if you don't want it.

After the story finishes, the experience **automatically returns to the
start screen** and waits. For the next visitor, just press **Space** again.

---

## Keys you might need

| Key | What it does |
|---|---|
| **Space** | Start — and pause / un-pause at any time |
| **S** | Skip the tutorial, the introduction, or the ending credits |
| **R** | Restart from the very beginning (for the next visitor) — it pauses on the start screen, press Space to begin |
| **↑ / ↓ arrows** (while paused) | Volume up / down (you can also drag the on-screen slider with the mouse) |
| **C** | Turn the on-screen captions (subtitles) on or off |
| **F** | Switch between fullscreen and a window |
| **Esc** | Quit the program completely |

Visitors don't need any keys — they only use their hands and voice.

Staff voice shortcuts: saying **"skip"** out loud works like the S key
during the tutorial, the introduction, and the ending.

---

## If something isn't working

**Everything freezes but the sound keeps playing (usually only the first run)**
→ This is the program building its speed-up files for a scene it hasn't shown
before. Press **Esc**, close it, and open `BHR.exe` again — it will pick up
where the files left off. Running **`Prewarm cache (run once).bat`** beforehand
(see setup) prevents this entirely.

**"Not responding" for a minute or two (usually only the first run)**
→ Same cause — it's busy building speed-up files. Wait one or two minutes and it
resumes on its own. If it's been longer than that, press **Esc** and reopen.

**The picture is frozen or the program is acting strange**
→ Press **Esc** to close it, then double-click `BHR.exe` again. This fixes
almost everything and takes under a minute.

**"It doesn't see me" — gestures aren't being picked up**
→ Press Esc and restart the program, then look at the camera check screen:
make sure the *whole person* is in view (head to feet) and the stick figure
turns green. Move the camera or the visitor's standing spot if needed.
Strong light shining straight into the camera can also blind it.

**No sound**
→ Check the speakers are on and plugged in, and the computer volume isn't
muted. Then pause with **Space** and raise the volume with the **↑** key.

**Saying words does nothing ("freedom", "go", "north"...)**
→ Check the microphone is plugged in. If it was plugged in *after* the
program started, close with **Esc** and start `BHR.exe` again.

**The camera was unplugged**
→ Plug it back in, then close with **Esc** and start `BHR.exe` again.
The program can't recover a camera that went away while it was running.

**A visitor is stuck on a step**
→ Nothing to do — every step moves on by itself after a while. If you want
to push it along, press **Space** twice (pause and un-pause won't skip
anything) or just let it time out.

---

## Turning it off at the end of the day

Press **Esc**. The window closes and nothing else needs to be done.
It's safe to shut the computer down normally after that.

---

## Good to know

- The experience **loops on its own**: story ends → start screen → press
  Space for the next visitor.
- The very first run on a new computer may **stutter slightly** in a few
  scenes while it builds its speed-up files. This clears up after the first
  couple of runs — the files are saved and reused.
- Those speed-up files grow inside the `BHR` folder over time and can get
  large. That's expected — the exhibition computer has plenty of room.
- The black text window behind the experience shows technical messages.
  You can ignore it, but don't close it — closing it closes the show.
