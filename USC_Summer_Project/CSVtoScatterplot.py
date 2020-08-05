import csv
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import cv2
import os


def main():
    all_xcords = []
    all_ycords = []
    notdone = True
    count = 0

    while notdone:
        if os.path.isfile('CSVLunarV2/gen_{:06d}.csv'.format(count-1)):
            print(count)
            NotDone = False
        data = pd.read_csv('CSVLunarV2/gen_{:06d}.csv'.format(count))
        # data = data.pivot("x", "y", "fitness")

        ax = sns.scatterplot(x='x', y='y', data=data)
        plt.xlim(-3,3)
        plt.ylim(-3,3)



    image_folder = 'LunarImages2'
    video_name = 'video4.avi'

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