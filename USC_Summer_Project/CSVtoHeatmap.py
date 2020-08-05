import csv
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import cv2
import os


def main():
    NotDone = True
    count = 100000
    sns.set_context('talk')

    for i in range(1):
        count += 0
        # if os.path.isfile('CSVLunarV2/gen_{:06d}.csv'.format(count-1)):
        #     print(count)
        #     NotDone = False
        # data = pd.read_csv('CSVLunarME/gen_{:06d}.csv'.format(count))
        data = pd.read_csv('CSVEnvironment/final.csv')
        data = data.pivot("y velocity", "x position", "avg fitness")
        # print(e)
        f, ax = plt.subplots(figsize=(8, 8))
        sns.heatmap(data, annot=False, ax=ax, xticklabels=12, yticklabels=12, vmin=-300, vmax=300, square=True)
        # plt.xlim(0, 100)
        # plt.ylim(0, 100)
        plt.savefig("LunarImages/final.png".format(count), dpi=300)
        # plt.savefig("LunarImages/genME_{:06d}.png".format(count), dpi=300)
        print('saving figure ', count)
        plt.close()

    image_folder = 'LunarImages'
    video_name = 'video3.avi'

    images = [img for img in os.listdir(image_folder) if img.endswith(".png")]
    print(images)
    frame = cv2.imread(os.path.join(image_folder, images[0]))
    height, width, layers = frame.shape

    video = cv2.VideoWriter(video_name, 0, 1, (width, height))

    for image in images:
        video.write(cv2.imread(os.path.join(image_folder, image)))

    cv2.destroyAllWindows()
    video.release()


if __name__ == '__main__':
    main()