import sys
import yaml
from pathlib import Path
import time
import argparse
import pygame
from ant import NFRealtime

def show_image_until_space(image_path):
    pygame.init()
    screen = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)
    image = pygame.image.load(image_path)
    image = pygame.transform.scale(image, screen.get_size())
    clock = pygame.time.Clock()

    waiting = True
    while waiting:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    waiting = False

        screen.blit(image, (0, 0))
        pygame.display.flip()

        clock.tick(60)

    pygame.display.quit()
    pygame.quit()
    time.sleep(10) # change this there

if __name__ == "__main__":

    with open("config_master.yml", "r") as f:
        config = yaml.safe_load(f)

    ## show instruction img
    imgs_dir = config.get("imgs_dir", "./")
    image_path = Path(imgs_dir) / "img_01.jpg"
    # show_image_until_space(image_path)

    ## now connect to stream and record RS EEG
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject_id", required=True)
    parser.add_argument("--visit", required=True)
    args = parser.parse_args()

    subject_id = args.subject_id
    visit = int(args.visit)
    subjects_dir = config.get("subjects_dir", "./")

    baseline_duration = config.get("baseline_duration")
    kwargs = {
            "subject_id": subject_id,
            "visit": visit,
            "subjects_dir": Path(subjects_dir),
            "montage": "easycap-M1",
            "mri": False,
            "artifact_correction": False,
            "verbose": False
            }
    nf = NFRealtime(session="baseline", **kwargs)
    
    # Connect to a mock LSL stream (we are using our simulated data)
    fname = "/Users/payamsadeghishabestari/ANT/data/simulated/pericalcarine-lh_10_2-raw.fif"
    nf.connect_to_lsl(mock_lsl=True, fname=fname)
    time.sleep(4)
    nf.record_baseline(baseline_duration=baseline_duration)

