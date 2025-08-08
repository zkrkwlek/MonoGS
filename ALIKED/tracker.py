import copy
import cv2
import numpy as np
import torch

class SimpleTracker(object):
    def __init__(self):
        self.pts_prev = None
        self.desc_prev = None

    def match(self, desc1, desc2):
        matches = self.mnn_mather(desc1, desc2)
        return matches

    def visualize2(self, img, pts1, pts2, delay = 0, save = False, filename = None):
        out = copy.deepcopy(img)
        points1 = pts1.detach().cpu().numpy()
        points2 = pts2.detach().cpu().numpy()
        for pt1, pt2 in zip(points1, points2):
            p1 = (int(round(pt1[0])), int(round(pt1[1])))
            p2 = (int(round(pt2[0])), int(round(pt2[1])))

            cv2.line(out, p1, p2, (0, 255, 0), 2, lineType=16)
            cv2.circle(out, p1, 1, (0, 0, 255), -1, lineType=16)
            cv2.circle(out, p2, 1, (255, 0, 0), -1, lineType=16)

        if save :
            cv2.imwrite(filename, out)
        else:
            cv2.imshow("asdfasdfasdf",out)
            cv2.waitKey(delay)


    def visualize(self, img1, img2, matches, pts1, pts2, mode=False):
        mpts1, mpts2 = pts1[matches[:, 0]], pts2[matches[:, 1]]#: or ...

        h1, w1,c1 = img1.shape
        h2, w2,c2 = img2.shape

        if mode:
            out_img = np.zeros((max(h1,h2), w1+w2, 3), dtype=np.uint8)
            out_img[:h1, :w1, :] = img1
            out_img[:h2, w1:w1+w2, :] = img2

            for i, (pt1, pt2) in enumerate(zip(mpts1, mpts2)):
                if i % 10 == 0:
                    p1 = (int(round(pt1[0])), int(round(pt1[1])))
                    p2 = (int(round(pt2[0])+w1), int(round(pt2[1])))
                    cv2.line(out_img, p1, p2, (0, 255, 0), lineType=16)
                    #cv2.circle(out_img, p1, 1, (0, 0, 255), -1, lineType=16)
                    #cv2.circle(out_img, p2, 1, (0, 0, 255), -1, lineType=16)
            for i, (pt1, pt2) in enumerate(zip(mpts1, mpts2)):
                if i % 10 == 0:
                    p1 = (int(round(pt1[0])), int(round(pt1[1])))
                    p2 = (int(round(pt2[0])+w1), int(round(pt2[1])))
                    cv2.circle(out_img, p1, 1, (0, 0, 255), -1, lineType=16)
                    cv2.circle(out_img, p2, 1, (0, 0, 255), -1, lineType=16)
        else:
            out_img = np.zeros((h1 + h2, max(w1, w2),3), dtype=np.uint8)
            out_img[:h1, :w1, :] = img1
            out_img[h1:h1 + h2, :w2, :] = img2

            for pt1, pt2 in zip(mpts1, mpts2):
                p1 = (int(round(pt1[0])), int(round(pt1[1])))
                p2 = (int(round(pt2[0])), int(round(pt2[1])+h1))
                cv2.line(out_img, p1, p2, (0, 255, 0), lineType=16)
                cv2.circle(out_img, p1, 1, (0, 0, 255), -1, lineType=16)
                cv2.circle(out_img, p2, 1, (0, 0, 255), -1, lineType=16)
        return out_img
    def update(self, img, pts, desc):
        N_matches = 0
        if self.pts_prev is None:
            self.pts_prev = pts
            self.desc_prev = desc

            out = copy.deepcopy(img)
            for pt1 in pts:
                p1 = (int(round(pt1[0])), int(round(pt1[1])))
                cv2.circle(out, p1, 1, (0, 0, 255), -1, lineType=16)
        else:
            matches = self.mnn_mather(self.desc_prev, desc)
            mpts1, mpts2 = self.pts_prev[matches[:, 0]], pts[matches[:, 1]]
            N_matches = len(matches)

            out = copy.deepcopy(img)
            for pt1, pt2 in zip(mpts1, mpts2):
                p1 = (int(round(pt1[0])), int(round(pt1[1])))
                p2 = (int(round(pt2[0])), int(round(pt2[1])))
                cv2.line(out, p1, p2, (0, 255, 0), lineType=16)
                cv2.circle(out, p2, 1, (0, 0, 255), -1, lineType=16)

            self.pts_prev = pts
            self.desc_prev = desc

        return out, N_matches

    def mnn_mather(self, desc1, desc2):
        sim = desc1 @ desc2.transpose()
        sim[sim < 0.9] = 0
        nn12 = np.argmax(sim, axis=1)
        nn21 = np.argmax(sim, axis=0)
        ids1 = np.arange(0, sim.shape[0])
        mask = (ids1 == nn21[nn12])
        matches = np.stack([ids1[mask], nn12[mask]])
        return matches.transpose()