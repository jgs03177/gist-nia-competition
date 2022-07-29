import argparse

parser = argparse.ArgumentParser()
parser.add_argument("frm", type=str, help="target csv file generated from evaluate.py")
parser.add_argument("to", type=str, help="converted csv file to be submitted to kaggle")
args = parser.parse_args()

file_from = args.frm
file_to = args.to

class_labels =['1', '2', '3', '4', '5', '7', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18', 
                    '19', '20', '21', '22', '23', '26', '27', '28', '29', '30', '31', '32', '33', '34', '36', 
                    '37', '38', '39', '40', '41', '42', '43', '44', '45', '46', '47', '48', '49', '50', '51', 
                    '52', '53', '54', '55', '56', '57', '58', '59', '60', '61', '62', '63', '64', '65', '66', 
                    '67', '68', '69', '70', '71', '72', '73', '74', '75', '76', '77', '78', '79', '80', '81', 
                    '82', '83', '85', '86', '87', '88', '89', '90', '91', '92', '93', '95', '96', '97', '98', 
                    '99', '100', '101', '102', '103', '104', '105', '106', '107', '108', '109', '110', '111', 
                    '113', '114', '115', '116', '117', '118', '119', '120', '121', '122', '123', '124', '125', 
                    '126', '127', '128', '129', '130', '131', '132', '133', '134', '135', '136']
                    
inv_class_labels = {}
for i,e in enumerate(class_labels):
    inv_class_labels[e]=i
    
a = open(file_from, "r")
*a2, = a
a.close()

b = open(file_to, "w")
b.write('id,pred\n')
for i,a in enumerate(a2[1:]):
    _,c2 = a.strip().split(',')
    b.write(f'S{i},{inv_class_labels[c2]}\n')
b.close()