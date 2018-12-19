import numpy as np
import json
import cv2
import move
import info
import vis_tool

if __name__ == '__main__':

    preprocess_json = False
    if preprocess_json:
        info.sort_json('/home/pzq/warp_data/pose/dance2/alphapose-results.json',
                       '/home/pzq/warp_data/pose/dance2/dance_pose.json')

    file = open('/home/pzq/warp_data/pose/dance2/dance_pose.json', 'r')
    people = json.load(file)
    num_people = len(people)

    syn_continuous = True

    if syn_continuous:
        for i, person in enumerate(people):

            if i == num_people - 1:
                continue


            person0 = person
            person1 = people[i + 1]

            tgt_name = 'syn' + person1['id']

            #move.warp_combine(person0, person1, '/home/pzq/warp_data/img/dance2/',
            #                     '/home/pzq/pose_warp/part_move/vis_data/dance2/debug/out/')

            move.warp_img_pair(person0, person1, '/home/pzq/warp_data/img/dance2/',
                             '/home/pzq/pose_warp/part_move/vis_data/dance2/dance2_syn_tgtmask/')

            if i % 10 == 0:
                print("Image {id} end".format(id=i))

    vis_key = False

    if vis_key:
        for i, person in enumerate(people):

            vis_tool.vis_joints(person, '/home/pzq/warp_data/img/dance2/',
                                '/home/pzq/pose_warp/part_move/vis_data/dance2/debug/')

            if i % 100 == 0:
                print("Image {id} ends".format(id = i))



