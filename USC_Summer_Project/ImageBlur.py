import operator
import cv2
import PIL
import operator
from PIL import Image
from PIL import ImageDraw


def create_mask(img):
    mask = Image.new('L', img.size)
    pixels = mask.load()
    for i in range(img.size[0]):
        for j in range(img.size[1]):
            px = img.getpixel((i, j))
            values = sum(px)
            grey = 100 if values >= 100 * 3 else 0
            pixels[i, j] = (grey,)
    # mask.show()
    return mask


def main():
    count = 0
    composite_image = None
    vidcap = cv2.VideoCapture('recording/vid.mp4')
    while vidcap.isOpened():
        hasframe, frame = vidcap.read()
        if hasframe:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            im_pil = Image.fromarray(frame)
            if composite_image:
                mask = create_mask(im_pil)
                composite_image.paste(im_pil, (0, 0), mask)
            else:
                composite_image = im_pil
            print(count)
            count += 15
            vidcap.set(1, count)
        else:
            vidcap.release()
            break

    composite_image.save('motion_blur.png')


if __name__ == '__main__':
    main()
