import os
import cv2
import imageio

video_file = "lenia.mp4"  # The path to your video file or GIF
output_directory = "downloaded_video_frames"  # Desired output folder
batch_size = 32  # Number of frames to process in each batch


def video_to_images(video_path, output_folder, batch_size=32):
  """
  Converts a video or GIF to a sequence of images in batches.

  Args:
    video_path: Path to the video or GIF file.
    output_folder: Path to the folder where images will be saved.
    batch_size: Number of frames to process in each batch.
  """

  try:
    # Create the output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Determine if it's a video or GIF
    if video_path.endswith(".gif"):
      reader = imageio.get_reader(video_path)
      num_frames = reader.get_meta_data()['nframes']

      for i in range(0, num_frames, batch_size):
        batch_frames = [reader.get_data(j) for j in range(i, min(i + batch_size, num_frames))]
        for k, frame in enumerate(batch_frames):
          cv2.imwrite(os.path.join(output_folder, f"{i + k:04d}.png"), frame)
        print(f"Processed frames {i} to {min(i + batch_size, num_frames)}")
    else:  # Assume it's a video
      # Open the video using OpenCV
      vidcap = cv2.VideoCapture(video_path)
      success, image = vidcap.read()
      count = 0

      while success:
        # Save the current frame as an image
        cv2.imwrite(os.path.join(output_folder, f"{count:04d}.png"), image)  # Use f-string for formatting, leading zeros for better sorting
        success, image = vidcap.read()
        count += 1
        if count % batch_size == 0:
            print(f"Processed {count} frames")

      print(f"Video '{video_path}' successfully converted to images in '{output_folder}'")

  except FileNotFoundError:
    print(f"Error: Video or GIF file '{video_path}' not found.")
  except Exception as e:
    print(f"An error occurred: {e}")


if __name__ == "__main__":

  video_to_images(video_file, output_directory, batch_size) 
