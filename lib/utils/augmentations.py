import torch
from torchvision import transforms
import cv2
import numpy as np
import types
from numpy import random
from collections import defaultdict

def cv2_demo1(frame, detections):
    det = []
    COLORS = [(255, 0, 0)]
    for i in range(detections.shape[0]):
        pt = detections[i, :]
        cv2.rectangle(frame,(int(pt[0])-4, int(pt[1])-4),(int(pt[2])+4, int(pt[3])+4),COLORS[0], 2)

    return frame

def intersect(box_a, box_b):
    max_xy = np.minimum(box_a[:, 2:], box_b[2:])
    min_xy = np.maximum(box_a[:, :2], box_b[:2])
    inter = np.clip((max_xy - min_xy), a_min=0, a_max=np.inf)
    return inter[:, 0] * inter[:, 1]


def jaccard_numpy(box_a, box_b):
    """Compute the jaccard overlap of two sets of boxes.  The jaccard overlap
    is simply the intersection over union of two boxes.
    E.g.:
        A ∩ B / A ∪ B = A ∩ B / (area(A) + area(B) - A ∩ B)
    Args:
        box_a: Multiple bounding boxes, Shape: [num_boxes,4]
        box_b: Single bounding box, Shape: [4]
    Return:
        jaccard overlap: Shape: [box_a.shape[0], box_a.shape[1]]
    """
    inter = intersect(box_a, box_b)
    area_a = ((box_a[:, 2]-box_a[:, 0]) *
              (box_a[:, 3]-box_a[:, 1]))  # [A,B]
    area_b = ((box_b[2]-box_b[0]) *
              (box_b[3]-box_b[1]))  # [A,B]
    union = area_a + area_b - inter
    return inter / union  # [A,B]


class Compose(object):
    """Composes several augmentations together.
    Args:
        transforms (List[Transform]): list of transforms to compose.
    Example:
        >>> augmentations.Compose([
        >>>     transforms.CenterCrop(10),
        >>>     transforms.ToTensor(),
        >>> ])
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, img_fg, boxes, labels, ids, pre_boxes, pre_ids):
        for t in self.transforms:
            img, img_fg, boxes, labels, ids, pre_boxes, pre_ids = t(img, img_fg, boxes, labels, ids, pre_boxes, pre_ids)
        return img, img_fg, boxes, labels, ids, pre_boxes, pre_ids


class Lambda(object):
    """Applies a lambda as a transform."""

    def __init__(self, lambd):
        assert isinstance(lambd, types.LambdaType)
        self.lambd = lambd

    def __call__(self, img, boxes=None, labels=None):
        return self.lambd(img, boxes, labels)


class ConvertFromInts(object):
    def __call__(self, image, image_fg, boxes, labels, ids, pre_boxes, pre_ids):
        return image.astype(np.float32), image_fg.astype(np.float32), boxes, labels, ids, pre_boxes, pre_ids


class SubtractMeans(object):
    def __init__(self, mean):
        self.mean = np.array(mean, dtype=np.float32)

    def __call__(self, image, boxes=None, labels=None):
        image = image.astype(np.float32)
        image -= self.mean
        return image.astype(np.float32), boxes, labels


class ToAbsoluteCoords(object):
    def __call__(self, image, boxes=None, labels=None):
        height, width, channels = image.shape
        boxes[:, 0] *= width
        boxes[:, 2] *= width
        boxes[:, 1] *= height
        boxes[:, 3] *= height

        return image, boxes, labels


class ToPercentCoords(object):
    def __call__(self, image, boxes=None, labels=None):
        height, width, channels = image.shape
        boxes[:, 0] /= width
        boxes[:, 2] /= width
        boxes[:, 1] /= height
        boxes[:, 3] /= height

        return image, boxes, labels


class Resize(object):
    def __init__(self, size=300):
        self.size = size

    def __call__(self, image, boxes=None, labels=None):
        image = cv2.resize(image, (self.size,
                                 self.size))
        return image, boxes, labels


class RandomSaturation(object):
    def __init__(self, lower=0.5, upper=1.5):
        self.lower = lower
        self.upper = upper
        assert self.upper >= self.lower, "contrast upper must be >= lower."
        assert self.lower >= 0, "contrast lower must be non-negative."

    def __call__(self, image, image_fg, boxes, labels, ids, pre_boxes, pre_ids):
        if random.randint(2):
            image[:, :, 1] *= random.uniform(self.lower, self.upper)

        return image, image_fg, boxes, labels, ids, pre_boxes, pre_ids


class RandomHue(object):
    def __init__(self, delta=18.0):
        assert delta >= 0.0 and delta <= 360.0
        self.delta = delta

    def __call__(self, image, image_fg, boxes, labels, ids, pre_boxes, pre_ids):
        if random.randint(2):
            image[:, :, 0, :] += random.uniform(-self.delta, self.delta)
            image[:, :, 0, :][image[:, :, 0, :] > 360.0] -= 360.0
            image[:, :, 0, :][image[:, :, 0, :] < 0.0] += 360.0
        return image, image_fg, boxes, labels, ids, pre_boxes, pre_ids


class RandomLightingNoise(object):
    def __init__(self):
        self.perms = ((0, 1, 2), (0, 2, 1),
                      (1, 0, 2), (1, 2, 0),
                      (2, 0, 1), (2, 1, 0))

    def __call__(self, image, image_fg, boxes, labels, ids, pre_boxes, pre_ids):
        if random.randint(2):
            swap = self.perms[random.randint(len(self.perms))]
            shuffle = SwapChannels(swap)  # shuffle channels
            image = shuffle(image)
        return image, image_fg, boxes, labels, ids, pre_boxes, pre_ids


class ConvertColor(object):
    def __init__(self, current='BGR', transform='HSV'):
        self.transform = transform
        self.current = current

    def __call__(self, image, image_fg, boxes, labels, ids, pre_boxes, pre_ids):
        Num = image.shape
        if (len(Num) == 3):
            if self.current == 'BGR' and self.transform == 'HSV':

                image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            elif self.current == 'HSV' and self.transform == 'BGR':
                image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)
            else:
                raise NotImplementedError
        else:
            for ii in range(Num[-1]):
                if self.current == 'BGR' and self.transform == 'HSV':
                    image[:,:,:,ii] = cv2.cvtColor(np.squeeze(image[:,:,:,ii]), cv2.COLOR_BGR2HSV)
                elif self.current == 'HSV' and self.transform == 'BGR':
                    image[:,:,:,ii] = cv2.cvtColor(np.squeeze(image[:,:,:,ii]), cv2.COLOR_HSV2BGR)
                else:
                    raise NotImplementedError
        return image, image_fg, boxes, labels, ids, pre_boxes, pre_ids


class RandomContrast(object):
    def __init__(self, lower=0.9, upper=1.1):
        self.lower = lower
        self.upper = upper
        assert self.upper >= self.lower, "contrast upper must be >= lower."
        assert self.lower >= 0, "contrast lower must be non-negative."

    # expects float image
    def __call__(self, image, image_fg, boxes, labels, ids, pre_boxes, pre_ids):
        if random.randint(2):
            alpha = random.uniform(self.lower, self.upper)
            image *= alpha
        return image, image_fg, boxes, labels, ids, pre_boxes, pre_ids


class RandomBrightness(object):
    def __init__(self, delta=32):
        assert delta >= 0.0
        assert delta <= 255.0
        self.delta = delta

    def __call__(self, image, image_fg, boxes, labels, ids, pre_boxes, pre_ids):
        if random.randint(2):
            delta = random.uniform(-self.delta, self.delta)
            image += delta / 255.0 * 6
        return image, image_fg, boxes, labels, ids, pre_boxes, pre_ids


class ToCV2Image(object):
    def __call__(self, tensor, boxes=None, labels=None):
        return tensor.cpu().numpy().astype(np.float32).transpose((1, 2, 0)), boxes, labels


class ToTensor(object):
    def __call__(self, cvimage, boxes=None, labels=None):
        return torch.from_numpy(cvimage.astype(np.float32)).permute(2, 0, 1), boxes, labels


class RandomSampleCrop(object):
    """Crop
    Arguments:
        img (Image): the image being input during training
        boxes (Tensor): the original bounding boxes in pt form
        labels (Tensor): the class labels for each bbox
        mode (float tuple): the min and max jaccard overlaps
    Return:
        (img, boxes, classes)
            img (Image): the cropped image
            boxes (Tensor): the adjusted bounding boxes in pt form
            labels (Tensor): the class labels for each bbox
    """
    def __init__(self):
        self.sample_options = (
            # using entire original input image
            None,
            # sample a patch s.t. MIN jaccard w/ obj in .1,.3,.4,.7,.9
            (0.1, None),
            (0.3, None),
            (0.7, None),
            (0.9, None),
            # randomly sample a patch
            (None, None),
        )

    def __call__(self, image, boxes=None, labels=None):
        height, width, _ = image.shape
        while True:
            # randomly choose a mode
            mode = random.choice(self.sample_options)
            if mode is None:
                return image, boxes, labels

            min_iou, max_iou = mode
            if min_iou is None:
                min_iou = float('-inf')
            if max_iou is None:
                max_iou = float('inf')

            # max trails (50)
            for _ in range(50):
                current_image = image

                w = random.uniform(0.3 * width, width)
                h = random.uniform(0.3 * height, height)

                # aspect ratio constraint b/t .5 & 2
                if h / w < 0.5 or h / w > 2:
                    continue

                left = random.uniform(width - w)
                top = random.uniform(height - h)

                # convert to integer rect x1,y1,x2,y2
                rect = np.array([int(left), int(top), int(left+w), int(top+h)])

                # calculate IoU (jaccard overlap) b/t the cropped and gt boxes
                overlap = jaccard_numpy(boxes, rect)

                # is min and max overlap constraint satisfied? if not try again
                if overlap.min() < min_iou and max_iou < overlap.max():
                    continue

                # cut the crop from the image

                current_image = current_image[rect[1]:rect[3], rect[0]:rect[2],
                                              :]

                # keep overlap with gt box IF center in sampled patch
                centers = (boxes[:, :2] + boxes[:, 2:]) / 2.0

                # mask in all gt boxes that above and to the left of centers

                m1 = (rect[0] < centers[:, 0]) * (rect[1] < centers[:, 1])


                # mask in all gt boxes that under and to the right of centers
                m2 = (rect[2] > centers[:, 0]) * (rect[3] > centers[:, 1])


                # mask in that both m1 and m2 are true
                mask = m1 * m2


                # have any valid boxes? try again if not
                if not mask.any():
                    continue


                # take only matching gt boxes
                current_boxes = boxes[mask, :].copy()


                # take only matching gt labels
                current_labels = labels[mask]


                # should we use the box left and top corner or the crop's
                current_boxes[:, :2] = np.maximum(current_boxes[:, :2],
                                                  rect[:2])
                # adjust to crop (by substracting crop's left,top)
                current_boxes[:, :2] -= rect[:2]

                current_boxes[:, 2:] = np.minimum(current_boxes[:, 2:],
                                                  rect[2:])
                # adjust to crop (by substracting crop's left,top)
                current_boxes[:, 2:] -= rect[:2]

                return current_image, current_boxes, current_labels


class CropFixArea(object):
    """Crop
    Arguments:
        img (Image): the image being input during training
        boxes (Tensor): the original bounding boxes in pt form
        labels (Tensor): the class labels for each bbox
        mode (float tuple): the min and max jaccard overlaps
    Return:
        (img, boxes, classes)
            img (Image): the cropped image
            boxes (Tensor): the adjusted bounding boxes in pt form
            labels (Tensor): the class labels for each bbox
    """
    def __init__(self, opt):
        self.opt = opt
        self.sample_area = (512, 512)

    def gen_mask(self, boxes, rect):
        # keep overlap with gt box IF center in sampled patch
        centers = (boxes[:, :2] + boxes[:, 2:]) / 2.0
        # mask in all gt boxes that above and to the left of centers
        m1 = (rect[0] <= centers[:, 0]) * (rect[1] <= centers[:, 1])
        # mask in all gt boxes that under and to the right of centers
        m2 = (rect[2] >= centers[:, 0]) * (rect[3] >= centers[:, 1])
        # mask in that both m1 and m2 are true
        mask = m1 * m2
        return mask

    def crop(self, box, rect):
        # use the box left and top corner or the crop's
        box[:, :2] = np.maximum(box[:, :2], rect[:2])
        # adjust to crop (by substracting crop's left,top)
        box[:, :2] -= rect[:2]

        box[:, 2:] = np.minimum(box[:, 2:], rect[2:])
        # adjust to crop (by substracting crop's left,top)
        box[:, 2:] -= rect[:2]
        return box

    def __call__(self, image, image_fg, boxes, labels, ids, pre_boxes, pre_ids):
        height, width, _, seq_num = image.shape
        _, _, fg_cha, long_seq_num = image_fg.shape
        while True:
            if height < self.sample_area[1] or width < self.sample_area[0]:
                image_pad = np.zeros([max(height, self.sample_area[1]), max(width, self.sample_area[0]), 3, seq_num])
                image_fg_pad = np.zeros([max(height, self.sample_area[1]), max(width, self.sample_area[0]), fg_cha, long_seq_num])
            else:
                image_pad = image
                image_fg_pad = image_fg
            if height < self.sample_area[1] or width < self.sample_area[0]:
                for i in range(long_seq_num):
                    image_fg_pad[:,:,:,i] = cv2.copyMakeBorder(image_fg[:,:,:,i], 0, max(0, self.sample_area[1] - height), 0, max(0, self.sample_area[0] - width), cv2.BORDER_CONSTANT, value=(0,0,0))
                    if i < seq_num:
                        image_pad[:,:,:,i] = cv2.copyMakeBorder(image[:,:,:,i], 0, max(0, self.sample_area[1] - height), 0, max(0, self.sample_area[0] - width), cv2.BORDER_CONSTANT, value=(0,0,0))
                height = max(height, self.sample_area[1])
                width = max(width, self.sample_area[0])
            
            flag = 'have_box'
            for count in range(37):
                current_image = image_pad
                current_fg = image_fg_pad

                w = self.sample_area[0]
                h = self.sample_area[1]
                if flag == 'have_box':
                    left = random.uniform(width - w)
                    top = random.uniform(height - h)
                else:
                    left = (width - w) / 5 * ((count - 1) % 6)
                    top = (height - h) / 5 * int((count - 1) / 6)
                    if (count - 1) % 6 == 5:
                        left = width - w
                    if int((count - 1) / 6) == 5:
                        top = height - h

                # convert to integer rect x1,y1,x2,y2
                rect = np.array([int(left), int(top), int(left+w), int(top+h)])

                # cut the crop from the image
                current_image = current_image[rect[1]:rect[3], rect[0]:rect[2], :, :]
                current_fg = current_fg[rect[1]:rect[3], rect[0]:rect[2], :, :]

                if boxes.shape[0] == 0:
                    print('Attention!!! Image has no box!!!')
                    return current_image, current_fg, boxes, labels, ids

                mask = self.gen_mask(boxes, rect)

                # have any valid boxes?
                if not mask.any() and count < 36:
                    flag = 'no_box'
                    continue
                if not mask.any() and count == 36:
                    raise ValueError('Strange thing!!! Image has no box!!!')

                # take only matching gt boxes
                current_boxes = boxes[mask, :].copy()
                current_boxes = self.crop(current_boxes, rect)

                # take only matching gt labels
                current_labels = labels[mask]
                current_ids = ids[mask]

                pre_mask = defaultdict(list)
                pre_boxes_crop = defaultdict(list)
                pre_ids_crop = defaultdict(list)
                for j in range(self.opt.seqLen - 1):
                    pre_mask[j + 1] = self.gen_mask(pre_boxes[j + 1], rect)
                
                for j in range(self.opt.seqLen - 1):
                    pre_boxes_crop[j + 1] = pre_boxes[j + 1][pre_mask[j + 1], :].copy()
                    pre_boxes_crop[j + 1] = self.crop(pre_boxes_crop[j + 1], rect)
                    pre_ids_crop[j + 1] = pre_ids[j + 1][pre_mask[j + 1]]

                return current_image, current_fg, current_boxes, current_labels, current_ids, pre_boxes_crop, pre_ids_crop


class Expand(object):
    def __init__(self, mean):
        self.mean = mean

    def __call__(self, image, boxes, labels):
        if random.randint(2):
            return image, boxes, labels

        height, width, depth = image.shape
        ratio = random.uniform(1, 4)
        left = random.uniform(0, width*ratio - width)
        top = random.uniform(0, height*ratio - height)
        expand_image = np.zeros(
            (int(height*ratio), int(width*ratio), depth),
            dtype=image.dtype)
        expand_image[:, :, :] = self.mean
        expand_image[int(top):int(top + height),
                     int(left):int(left + width)] = image
        image = expand_image

        boxes = boxes.copy()
        boxes[:, :2] += (int(left), int(top))
        boxes[:, 2:] += (int(left), int(top))

        return image, boxes, labels


class RandomMirror(object):
    def __init__(self, opt):
        self.opt = opt

    def __call__(self, image, image_fg, boxes, classes, ids, pre_boxes, pre_ids):
        if boxes.shape[0] == 0:
            return image, image_fg, boxes, classes, ids, pre_boxes, pre_ids
        for i in range(self.opt.seqLen - 1):
            if pre_boxes[i + 1].shape[0] == 0:
                return image, image_fg, boxes, classes, ids, pre_boxes, pre_ids
        imshape = image.shape
        if(len(imshape)==3):
            _, width, _ = image.shape
            if random.randint(3):
                image = image[:, ::-1]
                image_fg = image_fg[:, ::-1]
                boxes = boxes.copy()
                boxes[:, 0::2] = width - boxes[:, 2::-2]
                for i in range(self.opt.seqLen - 1):
                    pre_box = pre_boxes[i + 1].copy()
                    pre_boxes[i + 1][:, 0::2] = width - pre_box[:, 2::-2]
        else:
            height, width, _, _= image.shape
            if random.randint(2):
                image = image[:, ::-1,:,:]
                image_fg = image_fg[:, ::-1,:,:]
                boxes = boxes.copy()
                boxes[:, 0::2] = width - boxes[:, 2::-2]
                for i in range(self.opt.seqLen - 1):
                    pre_box = pre_boxes[i + 1].copy()
                    pre_boxes[i + 1][:, 0::2] = width - pre_box[:, 2::-2]
            if random.randint(2):
                image = image[::-1, :,:,:]
                image_fg = image_fg[::-1, :,:,:]
                boxes = boxes.copy()
                boxes[:, 1::2] = height - boxes[:, 3::-2]
                for i in range(self.opt.seqLen - 1):
                    pre_box = pre_boxes[i + 1].copy()
                    pre_boxes[i + 1][:, 1::2] = height - pre_box[:, 3::-2]

        return image, image_fg, boxes, classes, ids, pre_boxes, pre_ids


class SwapChannels(object):
    """Transforms a tensorized image by swapping the channels in the order
     specified in the swap tuple.
    Args:
        swaps (int triple): final order of channels
            eg: (2, 1, 0)
    """

    def __init__(self, swaps):
        self.swaps = swaps

    def __call__(self, image):
        """
        Args:
            image (Tensor): image tensor to be transformed
        Return:
            a tensor with channels swapped according to swap
        """
        # if torch.is_tensor(image):
        #     image = image.data.cpu().numpy()
        # else:
        #     image = np.array(image)
        image = image[:, :, self.swaps]
        return image


class PhotometricDistort(object):
    def __init__(self):
        self.rand_contrast = RandomContrast()
        self.rand_brightness = RandomBrightness()

    def __call__(self, image, image_fg, boxes, labels, ids, pre_boxes, pre_ids):
        im = image.copy()
        im, image_fg, boxes, labels, ids, pre_boxes, pre_ids = self.rand_brightness(im, image_fg, boxes, labels, ids, pre_boxes, pre_ids)
        im, image_fg, boxes, labels, ids, pre_boxes, pre_ids = self.rand_contrast(im, image_fg, boxes, labels, ids, pre_boxes, pre_ids)
        return im, image_fg, boxes, labels, ids, pre_boxes, pre_ids


class Augmentation(object):
    def __init__(self, opt, size=300, mean=(104, 117, 123)):
        self.mean = mean
        self.size = size
        self.augment = Compose([
            CropFixArea(opt),
            RandomMirror(opt),
            ConvertFromInts(),
        ])

    def __call__(self, img, img_fg, boxes, labels, ids, pre_boxes, pre_ids):
        return self.augment(img, img_fg, boxes, labels, ids, pre_boxes, pre_ids)
