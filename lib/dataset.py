"""
file for making up data sets
The goal is to make any json into id based format
Dataset includes:
    Market1501
"""
import json
import numpy as np

class Person(object):
    """
    This is the class for id information
    """
    def __init__(self, id, img_name, joint):
        """
        name: id of this person
        start: start of this id
        img_num: number of images
        """
        self.name = id
        self.imgs = [img_name]
        self.joints = [joint]
        self.img_num = 1


class Dataset(object):
    """
    This is the class for whole data set
    """
    def __init__(self):
        """
        people_num: number of people
        people: information about each person
        """
        self.people_num = 0
        self.people = []

    def make_data(self, json_path):
        """
        make json file grouped by person id
        """
        f = open(json_path, 'r')
        dic = json.load(f)
        img_num = len(dic)
        prev_id = -1
        prev_loc = -1

        for i, person in enumerate(dic):
            person_id = int(person['id'][:4])
            if person_id == prev_id:
                self.people[prev_loc].img_num += 1
                self.people[prev_loc].imgs.append(person['id'])
                self.people[prev_loc].joints.append(np.array(person['joints']))
            elif person_id != prev_id:
                tmp = Person(person_id, person['id'], np.array(person['joints']))
                self.people.append(tmp)

                prev_loc += 1
                prev_id = person_id
                self.people_num += 1
