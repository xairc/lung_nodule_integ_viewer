import numpy as np
import os

#data_dir = '/root/ssd_data/luna_segment_attribute/'
data_dir = './data/'

# attribute data index
#0 name
#1 pos_x
#2 pos_y
#3 pos_z
#4 size : < 5 samll, 5 < and 10 >, medium, 10 > big
#5 mal: 2.5 < benign, 2.5 < and 3.5 < neutral, 3.5 < malignant, range 1 ~ 5
#6 sphericiy: linear <-> ovoid <-> round, range 1 ~5
#7 margin: poorly defined <-> sharp, range 1 ~5
#8 spiculation: marked <-> None, range 1~5
#9 texture: nonsolid <-> part-solid <-> solid, range 1~5
#10 calcification: popcorn, laminated, solid, non-central, central, absent, 6 class
#11 internal_structure: soft tissue, fluid, fat, air, 4 class
#12 lobulation:marked <-> none, range 1 ~ 5
#13 subtlety: extreamely subtle <-> obvious,range 1 ~ 5
#14 hit_count


# check mal is benign, neutral, malignant
# 0 benign, 1 neutral, 2 malignant

def check_mal(mal):
    res = ''
    if (mal <= 2.5):
        res = 0
    elif (2.5 < mal <= 3.5):
        res = 1
    else:
        res = 2
    return res

#check rule and make sentneces
def rule(label, prob_list, size_rule_prob, rule_count_name, size_rule_count_name, attr_dict):
    size = label[4]
    mal = label[5]
    sphericity = label[6]
    margin = label[7]
    spiculation = label[8]
    texture = label[9]
    calcification = label[10]
    internal_structure = label[11]
    lobulation = label[12]
    subtlety = label[13]

    print ("#############################################")
    mal_res = check_mal(mal)

    #### check attribute has rule
    rule_idx = []
    size_idx = 0

    #benign rule check
    if (mal_res == 0):
        print ("mal is benign")
        if (lobulation <= 3):
            rule_idx.append(1)
        if (margin >= 3):
            rule_idx.append(2)
        if (sphericity >= 3):
            rule_idx.append(3)
        if (spiculation <= 2):
            rule_idx.append(4)
        if (subtlety <= 3):
            rule_idx.append(5)
        if (texture <= 3):
            rule_idx.append(6)
    #neutral rule check
    elif (mal_res == 1):
        print("mal is neutral")
        if (3 < lobulation and lobulation <= 4):
            rule_idx.append(1)
        if (3 > margin and lobulation >= 2):
            rule_idx.append(2)
        if (3 > sphericity and lobulation >= 2):
            rule_idx.append(3)
        if (2 < spiculation and lobulation <= 3):
            rule_idx.append(4)
        if (3 < subtlety and lobulation <= 4):
            rule_idx.append(5)
        if (3 < texture and lobulation <= 4):
            rule_idx.append(6)
    #malignant rule check
    elif (mal_res == 2):
        print("mal is malignant")
        if (calcification == 6):
            rule_idx.append(0)
        if (lobulation > 4):
            rule_idx.append(1)
        if (margin < 2):
            rule_idx.append(2)
        if (sphericity < 2):
            rule_idx.append(3)
        if (spiculation > 3):
            rule_idx.append(4)
        if (subtlety > 4):
            rule_idx.append(5)
        if (texture > 4):
            rule_idx.append(6)

    #check size rule
    if (size < 5):
        size_idx = 0
    if ( 5 <= size and size < 10 ):
        size_idx = 1
    if ( 10 < size):
        size_idx = 2

    prob_dict = {}

    #get size rule prob
    prob_dict[size_rule_count_name[size_idx]] = size_rule_prob[size_idx][mal_res]

    #get rule prob
    for i in range(len(rule_idx)):
        prob_dict[rule_count_name[rule_idx[i]]] = prob_list[mal_res][rule_idx[i]][mal_res]

    #sorting rule prob
    prob_dict_sort = sorted(prob_dict.items(), key=lambda kv: kv[1], reverse=True)

    #print top 3 rule sentence
    top_count = 0
    print('because', end=' ')
    for key, value in prob_dict_sort:
        top_count = top_count + 1
        print (attr_dict[mal_res][key], end=' ')

        if (key == 'medium' or key == 'small' or key == 'big'):
            print ( '('+str(size)+'mm)', end=' ')
        if (top_count >= 3):
            break
        else :
            print ('and', end=' ')
    print()

#count nodule per malignancy, size rule at dataset
#ex) size < 5 nodule all count, maliginant count, neutral count, benign count

def size_count_rule(label, size_rule_count_list):
    size = label[4]
    mal = label[5]

    mal_result = check_mal(mal)

    if (size < 5):
        size_rule_count_list[0][mal_result] = size_rule_count_list[0][mal_result] + 1
    if ( 5 <= size and size < 10 ):
        size_rule_count_list[1][mal_result] = size_rule_count_list[1][mal_result] + 1
    if ( 10 < size):
        size_rule_count_list[2][mal_result] = size_rule_count_list[2][mal_result] + 1

#count benign rule at dataset
def benign_count_rule(label, rule_count_list):
    size = label[4]
    mal = label[5]
    sphericity = label[6]
    margin = label[7]
    spiculation = label[8]
    texture = label[9]
    calcification = label[10]
    internal_structure = label[11]
    lobulation = label[12]
    subtlety = label[13]

    mal_result = check_mal(mal)

    if (lobulation <= 3):
        rule_count_list[1][mal_result] = rule_count_list[1][mal_result] + 1
    if (margin >= 3):
        rule_count_list[2][mal_result] = rule_count_list[2][mal_result] + 1
    if (sphericity >= 3):
        rule_count_list[3][mal_result] = rule_count_list[3][mal_result] + 1
    if (spiculation <= 2):
        rule_count_list[4][mal_result] = rule_count_list[4][mal_result] + 1
    if (subtlety <= 3):
        rule_count_list[5][mal_result] = rule_count_list[5][mal_result] + 1
    if (texture <= 3):
        rule_count_list[6][mal_result] = rule_count_list[6][mal_result] + 1

#count neutral rule at dataset
def neutral_count_rule(label, rule_count_list):
    size = label[4]
    mal = label[5]
    sphericity = label[6]
    margin = label[7]
    spiculation = label[8]
    texture = label[9]
    calcification = label[10]
    internal_structure = label[11]
    lobulation = label[12]
    subtlety = label[13]

    mal_result = check_mal(mal)

    if (3 < lobulation and lobulation <= 4):
        rule_count_list[1][mal_result] = rule_count_list[1][mal_result] + 1
    if ( 3 > margin and lobulation >= 2):
        rule_count_list[2][mal_result] = rule_count_list[2][mal_result] + 1
    if ( 3 > sphericity and lobulation >= 2):
        rule_count_list[3][mal_result] = rule_count_list[3][mal_result] + 1
    if (2 < spiculation and lobulation <= 3):
        rule_count_list[4][mal_result] = rule_count_list[4][mal_result] + 1
    if (3 < subtlety and lobulation <= 4):
        rule_count_list[5][mal_result] = rule_count_list[5][mal_result] + 1
    if (3 < texture and lobulation<= 4):
        rule_count_list[6][mal_result] = rule_count_list[6][mal_result] + 1

#count mal_count_rule at dataset
def mal_count_rule(label, rule_count_list):
    size = label[4]
    mal = label[5]
    sphericity = label[6]
    margin = label[7]
    spiculation = label[8]
    texture = label[9]
    calcification = label[10]
    internal_structure = label[11]
    lobulation = label[12]
    subtlety = label[13]

    mal_result = check_mal(mal)

    if (calcification == 6):
        rule_count_list[0][mal_result] = rule_count_list[0][mal_result] + 1
    if (lobulation > 4):
        rule_count_list[1][mal_result] = rule_count_list[1][mal_result] + 1
    if (margin < 2):
        rule_count_list[2][mal_result] = rule_count_list[2][mal_result] + 1
    if (sphericity < 2):
        rule_count_list[3][mal_result] = rule_count_list[3][mal_result] + 1
    if (spiculation > 3):
        rule_count_list[4][mal_result] = rule_count_list[4][mal_result] + 1
    if (subtlety > 4):
        rule_count_list[5][mal_result] = rule_count_list[5][mal_result] + 1
    if (texture > 4):
        rule_count_list[6][mal_result] = rule_count_list[6][mal_result] + 1
    # print (reason)

def print_label(label):
    print ("=========================================================================================================================================================")
    print ("name ", label[0], "size", label[4], "mal", label[5], "sphericity", label[6], "margin", label[7])
    print ("spiculation ", label[8], "texture", label[9], "calcification", label[10], "internal_structure", label[11], "lobulation", label[12], "subtlety", label[13])

#calc and print rule prob.
def calc_prob(mal_rule_count_list, neutral_rule_count_list, benign_rule_count_list, rule_count_name,
               mal_rule_prob, neutral_rule_prob, benign_rule_prob,
              size_rule_count_list, size_rule_count_name, size_rule_prob):
    print("==================================================")
    print("mal rule prob")
    for i in range(7):
        if (np.sum(mal_rule_count_list[i]) != 0):
            print(rule_count_name[i], " rule prob", end=' ')
            print("benign", mal_rule_count_list[i][0] / np.sum(mal_rule_count_list[i]), end=' ')
            print("neutral", mal_rule_count_list[i][1] / np.sum(mal_rule_count_list[i]), end=' ')
            print("mal", mal_rule_count_list[i][2] / np.sum(mal_rule_count_list[i]), end=' ')
            print("all count", np.sum(mal_rule_count_list[i]))
            mal_rule_prob[i] = mal_rule_count_list[i] / np.sum(mal_rule_count_list[i])
        else:
            print(rule_count_name[i], " does not have count")

    print("==================================================")
    print("neutral rule prob")
    for i in range(7):
        if (np.sum(neutral_rule_count_list[i]) != 0):
            print(rule_count_name[i], " rule prob", end=' ')
            print("benign", neutral_rule_count_list[i][0] / np.sum(neutral_rule_count_list[i]), end=' ')
            print("neutral", neutral_rule_count_list[i][1] / np.sum(neutral_rule_count_list[i]), end=' ')
            print("mal", neutral_rule_count_list[i][2] / np.sum(neutral_rule_count_list[i]), end=' ')
            print("all count", np.sum(neutral_rule_count_list[i]))
            neutral_rule_prob[i] = neutral_rule_count_list[i] / np.sum(neutral_rule_count_list[i])
        else:
            print(rule_count_name[i], " does not have count")

    print("==================================================")
    print("benign rule prob")
    for i in range(7):
        if (np.sum(benign_rule_count_list[i]) != 0):
            print(rule_count_name[i], " rule prob", end=' ')
            print("benign", benign_rule_count_list[i][0] / np.sum(benign_rule_count_list[i]), end=' ')
            print("neutral", benign_rule_count_list[i][1] / np.sum(benign_rule_count_list[i]), end=' ')
            print("mal", benign_rule_count_list[i][2] / np.sum(benign_rule_count_list[i]), end=' ')
            print("all count", np.sum(benign_rule_count_list[i]))
            benign_rule_prob[i] = benign_rule_count_list[i] / np.sum(benign_rule_count_list[i])
        else:
            print(rule_count_name[i], " does not have count")
    print("==================================================")
    print("size rule prob")
    for i in range(3):
        if (np.sum(size_rule_count_list[i]) != 0):
            print(size_rule_count_name[i], " rule prob", end=' ')
            print("benign", size_rule_count_list[i][0] / np.sum(size_rule_count_list[i]), end=' ')
            print("neutral", size_rule_count_list[i][1] / np.sum(size_rule_count_list[i]), end=' ')
            print("mal", size_rule_count_list[i][2] / np.sum(size_rule_count_list[i]), end=' ')
            print("all count", np.sum(size_rule_count_list[i]))
            size_rule_prob[i] = size_rule_count_list[i] / np.sum(size_rule_count_list[i])
        else:
            print(rule_count_name[i], " does not have count")


labels = []
nodule_labels = []

#list for count,
# 7attribute(calcification, lobulation, margin, sphericiy, spiculation, subtlety, texture)
# 3 malignamcy(benign, neutral, malignant)
mal_rule_count_list = np.zeros((7,3))
neutral_rule_count_list = np.zeros((7,3))
benign_rule_count_list = np.zeros((7,3))

mal_rule_prob = np.zeros((7,3))
neutral_rule_prob = np.zeros((7,3))
benign_rule_prob = np.zeros((7,3))

#rule count attribute key
rule_count_name = ['calcification', 'lobulation', 'margin', 'sphericity',
                   'spiculation', 'subtlety', 'texture']

#attribute rule explain word
mal_attr_dict = {'calcification': 'calcification is absent', 'lobulation': 'lobulation is none', 'margin': 'margin is Poorly defined',
                 'sphericity': 'sphericity is Linear', 'spiculation': 'spiculation is marked', 'subtlety': 'subtlety is obvious', 'texture': 'texture is solid',
                 'medium': 'size is medium', 'small': 'size is small', 'big': 'size is big'}
neutral_attr_dict = {'calcification': '', 'lobulation': 'lobulation is meidum', 'margin': 'margin is meidum',
                     'sphericity': 'sphericity is ovoid', 'spiculation': 'spiculation is meidum', 'subtlety': 'subtlety is subtle', 'texture': 'texture is part solid',
                     'medium': 'size is medium', 'small': 'size is small', 'big': 'size is big'}
benign_attr_dict = {'calcification': '', 'lobulation': 'lobulation is marked', 'margin': 'margin is poorly sharp',
                    'sphericity': 'sphericity is Round', 'spiculation': 'spiculation is marked', 'subtlety': 'subtlety is subtle', 'texture': 'texture is nonsolid',
                    'medium': 'size is medium', 'small': 'size is small', 'big': 'size is big'}

attr_dict = [benign_attr_dict, neutral_attr_dict, mal_attr_dict]

#list for size count
# 3 rule (small, medium, bing),
# 3 malignamcy(benign, neutral, malignant)a
size_rule_count_list = np.zeros((3,3))
#list for size prob
size_rule_prob = np.zeros((3,3))

size_rule_count_name = ['small', 'medium', 'big']

#load attribute
for idx in range(887):
    # l = np.load(os.path.join(data_dir, '%s_attribute.npy' % idx))

    num = str(idx)
    for i in range(3 - len(str(idx))):
        num = str('0') + num

    path = data_dir + num + '_attribute.npy'
    l = np.load(path)

    if np.all(l == 0):
        l = np.array([])
    labels.append(l)

for i, l in enumerate(labels):
    if len(l) > 0:
        for label in l:
            nodule_labels.append([np.concatenate([[i], label[1:]])])

nodule_labels = np.concatenate(nodule_labels, axis=0)

print (np.shape(nodule_labels))

#attribute score rounding
for i in range(np.shape(nodule_labels)[0]):
    nodule_labels[i][4] = round(nodule_labels[i][4], 2)
    nodule_labels[i][5] = round(nodule_labels[i][5], 2)
    nodule_labels[i][6] = round(nodule_labels[i][6], 2)
    nodule_labels[i][7] = round(nodule_labels[i][7], 2)
    nodule_labels[i][8] = round(nodule_labels[i][8], 2)
    nodule_labels[i][9] = round(nodule_labels[i][9], 2)
    nodule_labels[i][10] = round(nodule_labels[i][10], 2)
    nodule_labels[i][11] = round(nodule_labels[i][11], 2)
    nodule_labels[i][12] = round(nodule_labels[i][12], 2)
    nodule_labels[i][13] = round(nodule_labels[i][13], 2)

#count benign, neutral, malignant per rule
for i in range(np.shape(nodule_labels)[0]):
    #print_label(nodule_labels[i])
    mal_count_rule(nodule_labels[i], mal_rule_count_list)
    neutral_count_rule(nodule_labels[i], neutral_rule_count_list)
    benign_count_rule(nodule_labels[i], benign_rule_count_list)
    size_count_rule(nodule_labels[i], size_rule_count_list)

#calculate benign, neutral, malignancy prob
calc_prob(mal_rule_count_list, neutral_rule_count_list, benign_rule_count_list, rule_count_name,
            mal_rule_prob, neutral_rule_prob, benign_rule_prob,
            size_rule_count_list, size_rule_count_name, size_rule_prob)

prob_list = [benign_rule_prob, neutral_rule_prob, mal_rule_prob]
print (size_rule_prob)
print (benign_rule_prob)
print (neutral_rule_prob)
print (mal_rule_prob)

#print all noudle sentence
for i in range(np.shape(nodule_labels)[0]):
    rule(nodule_labels[i], prob_list, size_rule_prob, rule_count_name, size_rule_count_name, attr_dict)