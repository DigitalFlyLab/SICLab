import cv2
import numpy as np
import os
from PIL import Image


class VideoToNPZStimulus:

    def __init__(
        self,
        video_path,
        width=82,
        height=41,
        target_fps=1000,
        output_root="/video_stimulus/"
    ):
        self.video_path = video_path
        self.width = width
        self.height = height
        self.target_fps = target_fps
        self.output_root = output_root

        os.makedirs(self.output_root, exist_ok=True)

    def process_video(self):

        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            raise ValueError("cannot open video")

        src_fps = cap.get(cv2.CAP_PROP_FPS)

        repeat_factor = int(round(self.target_fps / src_fps))

        frames_list = []

        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # resize
            resized = cv2.resize(
                gray,
                (self.width, self.height),
                interpolation=cv2.INTER_AREA
            ).astype(np.uint8)

            for _ in range(repeat_factor):
                frames_list.append(resized)
                frame_idx += 1

        cap.release()

        frames = np.stack(frames_list, axis=-1)  # (height, width, T)
        print("Stimulus shape (H,W,T):", frames.shape)

        npz_path = os.path.join(self.output_root, "flybrid.npz")
        np.savez_compressed(npz_path, frames=frames)

        gif_path = os.path.join(self.output_root, "flybrid.gif")
        from PIL import Image
        gif_frames = [Image.fromarray(frames[:, :, i]) for i in range(frames.shape[2])]
        gif_frames[0].save(
            gif_path,
            save_all=True,
            append_images=gif_frames[1:],
            duration=1,
            loop=0
        )


def main():
    video_path = "./autofly.mp4"

    converter = VideoToNPZStimulus(
        video_path=video_path,
        width=82,
        height=41,
        target_fps=1000,
        output_root="./VideoNPZ/"
    )
    converter.process_video()


if __name__ == "__main__":
    main()