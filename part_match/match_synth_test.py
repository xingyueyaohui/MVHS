import numpy as np
import dataset
import match_policy as match
from part_move import move

if __name__ == '__main__':
    # make market data set
    market = dataset.Dataset()
    market.make_data('/home/pzq/warp_data/pose/market/market-pose.json')
    src_dir = '/home/pzq/warp_data/img/market/'
    dst_dir = '/home/pzq/pose_warp/part_match/vis_data/syn/'

    # randomly select several
    for i in range(market.people_num):
        if market.people[i].name == 2:
            person = market.people[i]

            for j in range(person.img_num):
                if j >= 20:
                    break
                # find the most likely sample
                match_res = match.euclid_match_human(person, j)
                id = match_res[0]['id']

                # warp the image
                info0 = {'id': person.imgs[id], 'joints': person.joints[id]}
                info1 = {'id': person.imgs[j], 'joints': person.joints[j]}

                move.warp_img_pair(info0, info1, src_dir, dst_dir)


        else:
            continue




