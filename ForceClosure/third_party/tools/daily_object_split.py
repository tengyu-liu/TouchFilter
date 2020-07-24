from collections import defaultdict
import random
import os
import json

base_dir = '/media/tengyu/data/shapenet/ShapeNetCore.v2'

taxonomy = os.path.join(base_dir, 'taxonomy.json')

taxonomy = json.load(open(taxonomy, 'r'))

test_split = {}
test_split['ShapeNetCore.v2'] = defaultdict(list)
train_split = {}
train_split['ShapeNetCore.v2'] = defaultdict(list)

pending = ['02876657']  # bottle
leaf = []

while len(pending) > 0:
    pending_len = len(pending)
    for taxo in taxonomy:
        if taxo['synsetId'] in pending:
            if len(taxo['children']) > 0:
                pending += taxo['children']
            leaf.append(taxo['synsetId'])
    pending = pending[pending_len:]

for taxo in taxonomy:
    synset_id = taxo['synsetId']
    # if taxo['name'].strip() in ["bottle", "beer bottle", "flask", "canteen", "jug", "mug", "beer mug,stein", "coffee mug", "phial,vial,ampule,ampul,ampoule", "pop bottle,soda bottle", "wine bottle", "bowl", "soup bowl", "can,tin,tin can", "beer can", "soda can", "guitar", "acoustic guitar", "helmet", "football helmet", "hard hat,tin hat,safety hat", "space helmet", "jar", "vase", "knife", "microphone,mike", "remote control,remote", "handset,French telephone", "radiotelephone,radiophone,wireless telephone", "cellular telephone,cellular phone,cellphone,cell,mobile phone"]:
    if synset_id in leaf:
        try:
            for model_id in os.listdir(os.path.join(base_dir, synset_id)):
                for syn in test_split.keys():
                    if model_id in test_split[syn]:
                        continue
                for syn in train_split.keys():
                    if model_id in train_split[syn]:
                        continue
                if random.random() < 0.1:
                    test_split['ShapeNetCore.v2'][synset_id].append(model_id)
                else:
                    train_split['ShapeNetCore.v2'][synset_id].append(model_id)
        except:
            continue

print('test', sum(len(test_split[x]) for x in test_split))
print('train', sum(len(train_split[x]) for x in train_split))
json.dump(test_split, open('daily_object_test.json', 'w'))
json.dump(train_split, open('daily_object_train.json', 'w'))
