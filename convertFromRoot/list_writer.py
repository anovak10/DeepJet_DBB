
import os
path = '/eos/user/a/anovak/train_files/'

f = open('train_list.txt', 'w')

for i in os.listdir(path):
	f.write(path+i+'\n')

f.close()


path = '/eos/user/a/anovak/test_files/'

f = open('test_list.txt', 'w')

for i in os.listdir(path):
	f.write(path+i+'\n')

f.close()
