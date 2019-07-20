import torch
import matplotlib.pyplot as plt
import torch.nn.functional as F
from mpl_toolkits.axes_grid1 import make_axes_locatable


def add_overhead(bbox, overhead, img_shape):
    '''
    1) accumulate all the computing overheads from all the residual layers
    2) add the overhead to the extra_fields

    :param bbox:
    :param overhead:
    :param img_shape:
    :return:
    '''
    prev = overhead[-1][0]
    for i in range(len(overhead) - 2, -1, -1):
        prev = F.interpolate(prev, size=overhead[i][0].shape[2:], mode='bilinear', align_corners=False)
        prev += overhead[i][0]

    # overlay_complexity: the accumulative computing overheads upon the whole image
    overlay_complexity = F.interpolate(prev, size=img_shape[:2], mode='bilinear', align_corners=False)
    overlay_complexity = overlay_complexity.reshape(overlay_complexity.shape[2:])

    bbox_overheads = []
    # calculate the computing overheads covered by each bbox
    bbox.bbox = bbox.bbox.int()
    for i in range(len(bbox.bbox)):
        x1, y1, x2, y2 = bbox.bbox[i]
        res = torch.mean(overlay_complexity.narrow(0, y1, y2 - y1).narrow(1, x1, x2 - x1))
        bbox_overheads.append(res)
    bbox.bbox = bbox.bbox.float()
    bbox.extra_fields['overheads'] = torch.stack(bbox_overheads).cpu().view(-1, 1)
    return bbox


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
            plt.savefig('ponder{}_{}.eps'.format(1, i), pad_inches=0, bbox_inches='tight')

        plt.show()
