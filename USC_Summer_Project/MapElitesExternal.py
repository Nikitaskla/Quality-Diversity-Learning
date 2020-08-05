import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import cv2
import os



num_params = 2



def eval_sphere(x):
    return np.sum(np.square(x-2.048))

#resolution, highbound, lowbound


def main():
    x = np.array([2.048] *num_params)
   # print(x)
   # print(eval_sphere(x))

    count = 0

    es = cma.CMAEvolutionStrategy(num_params * [0.0], 0.2, {"popsize": 30})
    while not es.stop():
        candidates = es.ask()
        print(candidates)

        fitnesses = [eval_sphere(x) for x in candidates]
        #print(fitnesses)

        temp = np.array(candidates)
        xcord = temp[:, 0]
        ycord = temp[:, 1]
        print(xcord)
        print(ycord)

        d = {}
        d["x"] = xcord
        d["y"] = ycord
        d = pd.DataFrame(d)

        ax = sns.scatterplot(x="x", y="y", data=d)
        plt.xlim(-5.12, 5.12)
        plt.ylim(-5.12, 5.12)
        plt.savefig("images/gen_{:03d}.png".format(count))
        plt.close()
        count+=1


        print(d)

        es.tell(candidates, fitnesses)

    image_folder = 'images'
    video_name = 'video.avi'

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