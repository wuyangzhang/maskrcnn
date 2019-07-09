import torch
import matplotlib.pyplot as plt
import torch.nn.functional as F
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.figure import figaspect

def ponder_cost_postproc(bbox, ponder_cost, img_shape):
    prev = ponder_cost[-1][0]
    for i in range(len(ponder_cost) - 2, -1, -1):
        # resize prev to the curr size and sum it up..
        prev = F.interpolate(prev, size=ponder_cost[i][0].shape[2:], mode='bilinear', align_corners=False)
        prev += ponder_cost[i][0]
    overlay_complexity = F.interpolate(prev, size=img_shape[:2], mode='bilinear', align_corners=False)
    overlay_complexity = overlay_complexity.reshape(overlay_complexity.shape[2:])
    complexity = []
    for i in range(len(bbox)):
        x1, y1, x2, y2 = bbox[i]
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        res = torch.mean(overlay_complexity.narrow(0, y1, y2 - y1).narrow(1, x1, x2 - x1))
        complexity.append(res.cpu())
    return complexity



def vis_ponder_cost(ponder_cost, is_write=True):
    i = 0
    for cost, _ in ponder_cost:
        fig, ax = plt.subplots()
        ax.set_yticklabels([])
        ax.set_xticklabels([])
        plt.gca().axes.get_yaxis().set_visible(False)
        plt.gca().axes.get_xaxis().set_visible(False)
        plt.gca().set_axis_off()
        plt.gca().xaxis.set_major_locator(plt.NullLocator())
        plt.gca().yaxis.set_major_locator(plt.NullLocator())
        plt.axis('off')
        ax.margins(0, 0)

        cost = cost.cpu()
        cost = cost.reshape(cost.shape[2:])
        im = ax.imshow(cost.cpu(), cmap='magma')
        divider = make_axes_locatable(ax)
        # cax = divider.append_axes("right", size="5%", pad=0.05)
        # plt.colorbar(im, cax=cax)

        fig.tight_layout()
        if is_write:
            i += 1
            plt.savefig('ponder{}_{}.eps'.format(1, i), pad_inches = 0 , bbox_inches='tight')

        plt.show()